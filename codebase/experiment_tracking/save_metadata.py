import os
import sys
from datetime import datetime
from pathlib import Path, PurePath

class ExperimentOrganizer():
    def __init__(self, root_path):
        self.root_path = root_path
        self.setup_folders()

    def setup_folders(self):
        self.codebase_path = Path(self.root_path+'/custom_code')
        self.states_path = Path(self.root_path+'/outputs/states/')
        self.logs_path = Path(self.root_path+'/outputs/logs/')
        
        self.codebase_path.mkdir(exist_ok=True, parents=True)
        self.states_path.mkdir(exist_ok=True, parents=True)
        Path(str(self.logs_path)+'/default/').mkdir(exist_ok=True, parents=True)

    def store_environment(self):
        os.system(f"pip freeze > {self.root_path}/requirements.txt")
        # os.system(f"mv requirements.txt {self.root_path}/env.txt")

    def store_yaml(self, source_file):
        os.system(f"cp {source_file} {self.root_path}/settings.yaml")

    def store_codebase(self, extensions):
        for ext in extensions:
            # os.system(f"cp --parents `find -wholename \*codebase/*{ext}` {self.codebase_path}/")
            os.system(f'cp --parents `find ./codebase/* -wholename "*{ext}"` {self.codebase_path}/')
        os.system(f"cp {sys.argv[0]} {self.codebase_path}/")


        