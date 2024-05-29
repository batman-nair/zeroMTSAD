from lightning.pytorch.callbacks import Callback

import os
import json

class SaveConfigCallback(Callback):
    def __init__(self, config: dict, run_info: dict):
        super().__init__()
        self.config = config
        self.run_info = run_info
        self.saved = False

    def _dump_files(self, log_dir: str):
        with open(os.path.join(log_dir, 'config.json'), 'w') as ff:
            json.dump(self.config, ff, indent=4)
        with open(os.path.join(log_dir, 'run.json'), 'w') as ff:
            json.dump(self.run_info, ff, indent=4)

    def on_train_start(self, trainer, pl_module):
        self.saved = True
        self._dump_files(trainer.logger.log_dir)

    def on_test_start(self, trainer, pl_module):
        if self.saved:
            return
        self._dump_files(trainer.logger.log_dir)
