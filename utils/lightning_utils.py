from lightning.pytorch.callbacks import Callback

import os
import json

class SaveConfigCallback(Callback):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def on_train_start(self, trainer, pl_module):
        with open(os.path.join(trainer.logger.log_dir, 'config.json'), 'w') as ff:
            json.dump(self.config, ff)
