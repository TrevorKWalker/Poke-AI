
import numpy as np
from gymnasium.spaces import Box, Space
from poke_env.data import GenData
from poke_env.player import Gen9EnvSinglePlayer
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env import Player

class CustomPokeEnv(Player):
    def embed_battle(self, battle: AbstractBattle):
        # Example embedding: HP difference and remaining pokémon difference
        return np.array([
            battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0,
            battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0,
            len(battle.available_switches),
            len(battle.opponent_team) - len(battle.available_switches)
        ])
    
    def compute_reward(self, battle: AbstractBattle):
        return (
            battle.opponent_active_pokemon.current_hp_fraction - battle.active_pokemon.current_hp_fraction
        )

class SimpleRLPlayer(Gen9EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2 , hp_value=1, victory_value=35.0,  status_value= .5, 
        )

    def embed_battle(self, battle: AbstractBattle):
        # Initialize arrays with default values
        moves_base_power = -np.ones(4, dtype=np.float32)  # -1 for unavailable moves
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)  # Default multiplier is 1

        # Process available moves
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 if move.base_power else 0  # Normalize power
            if move.type and battle.opponent_active_pokemon:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(9).type_chart,
                )

        # Pokémon HP fractions (normalize between 0 and 1)
        active_hp_fraction = (
            battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        )
        opponent_hp_fraction = (
            battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0
        )

        # Number of available switches (scaled by total team size)
        available_switches = len(battle.available_switches) / 6

        # Count fainted Pokémon (scaled by total team size)
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Current turn number (normalize by a reasonable max, e.g., 100)
        turn_normalized = battle.turn / 100.0

        # Remaining PP of available moves (normalized by max PP)
        moves_remaining_pp = np.zeros(4, dtype=np.float32)
        for i, move in enumerate(battle.available_moves):
            moves_remaining_pp[i] = move.current_pp / move.max_pp if move.max_pp else 0

        # Combine features into a single vector
        final_vector = np.concatenate(
            [
                moves_base_power,             # 4 values
                moves_dmg_multiplier,         # 4 values
                moves_remaining_pp,           # 4 values
                [active_hp_fraction, opponent_hp_fraction],  # 2 values
                [available_switches],         # 1 value
                [fainted_mon_team, fainted_mon_opponent],    # 2 values
                [turn_normalized],            # 1 value
            ]
        )

        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        # Define lower and upper bounds for each feature
        low = np.array(
            [-1] * 4 +  # moves_base_power
            [0] * 4 +   # moves_dmg_multiplier
            [0] * 4 +   # moves_remaining_pp
            [0, 0] +    # HP fractions
            [0] +       # available_switches
            [0, 0] +    # fainted Pokémon counts
            [0],        # turn_normalized
            dtype=np.float32
        )
        high = np.array(
            [1] * 4 +  # moves_base_power (normalized to max of 1)
            [4] * 4 +  # moves_dmg_multiplier (max type multiplier is 4)
            [1] * 4 +  # moves_remaining_pp
            [1, 1] +   # HP fractions
            [1] +      # available_switches
            [1, 1] +   # fainted Pokémon counts
            [1],       # turn_normalized
            dtype=np.float32
        )
        return Box(low, high, dtype=np.float32)

