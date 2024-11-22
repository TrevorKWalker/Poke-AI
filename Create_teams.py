import os

#This cycles through a directory and turns each file inside into a team
#params: directory - name of the path to folder
def create_teams(directory):
    teams = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                file_contents = file.read()
                teams.append(file_contents)
    return teams