from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.generative import GRUGMMVAE, GMMVAELoss, GMMVAEAnomalyDetector
from timesead_experiments.generative.vae.train_gmm_vae import get_training_pipeline, get_test_pipeline

import torch

class LitGRUGMMVAE(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = GRUGMMVAE(num_features, **model_params)
        self.loss = GMMVAELoss()
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = GMMVAEAnomalyDetector(self.model, **detector_params)


MODEL = LitGRUGMMVAE
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()

