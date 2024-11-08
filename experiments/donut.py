from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.generative import Donut, DonutAnomalyDetector, MaskedVAELoss
from timesead_experiments.generative.vae.train_donut import get_training_pipeline, get_test_pipeline

import torch

class LitDonut(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = Donut(num_features * seq_len, **model_params)
        self.loss = MaskedVAELoss()
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = DonutAnomalyDetector(self.model, **detector_params)


MODEL = LitDonut
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
