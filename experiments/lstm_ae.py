from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.reconstruction import LSTMAEMalhotra2016, LSTMAEAnomalyDetector
from timesead_experiments.reconstruction.train_lstm_ae import get_training_pipeline, get_test_pipeline

import torch

class LitLSTMAEMalhotra2016(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = LSTMAEMalhotra2016(num_features, **model_params)
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = LSTMAEAnomalyDetector(self.model)
        self.detector.fit(val_loader)


MODEL = LitLSTMAEMalhotra2016
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
