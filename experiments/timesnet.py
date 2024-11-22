from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.models.reconstruction import TimesNet
from timesead_experiments.reconstruction.train_timesnet import get_training_pipeline, get_test_pipeline

import torch

class LitTimesNet(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = TimesNet(seq_len, num_features, **model_params)
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = MSEReconstructionAnomalyDetector(self.model, **detector_params)


MODEL = LitTimesNet
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
