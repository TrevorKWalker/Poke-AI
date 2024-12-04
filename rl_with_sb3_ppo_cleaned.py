import numpy as np
from gymnasium.spaces import Box, Space
from poke_env.data import GenData
from poke_env.player import Gen8EnvSinglePlayer, RandomPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.environment.abstract_battle import AbstractBattle
#from poke_env.player.player import Player
#from poke_env.player.openai_api import OpenAIGymEnv
from poke_env.player.env_player import Gen8EnvSinglePlayer

MODEL_FILE = "poke_ppo_final.zip" #what the finished product will be
TENSORBOARD_LOG = "./ppo_tensorboard/"
SEED = 717 #Yvelta

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart = GenData.from_gen(8).type_chart
                )
                #moves_dmg_multiplier[i] = move.type.damage_multiplier(

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


# Define and check the environment
opponent = RandomPlayer(battle_format="gen8randombattle")
env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
check_env(env)

# Define Stable-Baselines3 model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    batch_size=32,
    tensorboard_log=TENSORBOARD_LOG,
    seed = SEED
)

# Train the model with checkpoints
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="poke_ppo_graph")
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model
model.save(MODEL_FILE)
