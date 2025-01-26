import torch
from torch.nn import functional as F

from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.prediction import LSTMPrediction, LSTMPredictionAnomalyDetector
from timesead_experiments.prediction.train_lstm_prediction_malhotra import get_training_pipeline, get_test_pipeline, get_batch_dim
from timesead.utils import torch_utils


class LitLSTMPrediction(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = LSTMPrediction(num_features, **model_params)
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = LSTMPredictionAnomalyDetector(self.model)
        self.detector.fit(val_loader)

    def on_test_start(self) -> None:
        super().on_test_start()
        self._errors = []
        self._labels = []
        self._counter = 0

    def test_step(self, batch, batch_idx):
        # Code derived from LSTMPredictionAnomalyDetector
        b_inputs, b_targets = batch
        pred = self.model(b_inputs)
        label, target = b_targets
        error = target - pred
        for j in range(error.shape[0]):
            for j2 in range(error.shape[1]):
                index = self._counter + j + j2
                if len(self._errors) <= index:
                    self._errors.append([])
                self._errors[index].append(error[j, j2])
        self._counter += error.shape[1]

        self._labels.append(label[-1].cpu())

    def on_test_end(self) -> None:
        # Code derived from LSTMPredictionAnomalyDetector
        self._errors = self._errors[self.model.prediction_horizon - 1:-self.model.prediction_horizon + 1]
        self._errors = torch_utils.nested_list2tensor(self._errors)
        self._errors = self._errors.view(self._errors.shape[0], -1)
        self._labels = torch.cat(self._labels, dim=0)
        labels = self._labels[:-self.model.prediction_horizon + 1]

        self._errors -= self.detector.mean
        scores = F.bilinear(self._errors, self._errors, self.detector.precision.unsqueeze(0)).squeeze(-1)
        scores = scores.cpu()

        assert labels.shape == scores.shape

        self._log_results(labels, scores)
        self._plot_anomalies(labels, scores)


MODEL = LitLSTMPrediction
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
BATCH_DIM = get_batch_dim()
