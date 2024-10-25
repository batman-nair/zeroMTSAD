from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.baselines.kmeans import KMeansAD
from timesead_experiments.baselines.train_kmeans import get_training_pipeline, get_test_pipeline

import torch

class LitKMeansAD(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params):
        super().__init__(run_params)
        self.model = KMeansAD(**model_params)
        self.model.window_size = seq_len if isinstance(seq_len, int) else seq_len[0]
        self.save_hyperparameters(model_params)
        self.model_trained = False

    def on_train_start(self):
        self.model_trained = True

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        data = inputs[0]
        data = data.reshape(data.shape[0], -1)
        self.model.model.partial_fit(data)

    def validation_step(self, batch, batch_idx):
        if not self.model_trained:  # To pass lightning sanity check
            return 0
        inputs, targets = batch
        scores = self.model.compute_online_anomaly_score(inputs)
        labels = self.model.format_online_targets(targets)
        loss, _ = self.metrics['best_ts_f1_score'](labels, scores)
        self.log('val_loss', loss, on_epoch=True)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = self.model

    def configure_optimizers(self):
        return None


MODEL = LitKMeansAD
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
