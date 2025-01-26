from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.models.reconstruction import FEDformer
from timesead_experiments.reconstruction.train_timesnet import get_training_pipeline, get_test_pipeline, get_batch_dim

import torch

class LitFedformer(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = FEDformer(seq_len, num_features, **model_params)
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = MSEReconstructionAnomalyDetector(self.model, **detector_params)


MODEL = LitFedformer
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
BATCH_DIM = get_batch_dim()
