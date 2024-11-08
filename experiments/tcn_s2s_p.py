from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.prediction import TCNS2SPrediction, TCNS2SPredictionAnomalyDetector
from timesead_experiments.prediction.train_tcn_prediction_he import get_training_pipeline, get_test_pipeline

import torch
from torch.nn import functional as F

class LitTCNS2SP(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = TCNS2SPrediction(num_features, **model_params)
        self.save_hyperparameters(model_params)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = TCNS2SPredictionAnomalyDetector(self.model, **detector_params)
        self.detector.fit(val_loader)

    def on_test_start(self) -> None:
        super().on_test_start()
        # Implementing detector code here are as it works over the whole dataset only
        self._errors = []
        self._labels = []
        self._counter = 0

    def test_step(self, batch, batch_idx):
        # Code derived from TCNS2SPredictionAnomalyDetector
        b_inputs, b_targets = batch
        pred = self.model(b_inputs)
        label, target = b_targets
        error = target[:, -self.detector.offset:] - pred[:, -self.detector.offset:]
        for j in range(error.shape[0]):
            for j2 in range(error.shape[1]):
                index = self._counter + j + j2
                if len(self._errors) <= index:
                    self._errors.append([])
                self._errors[index].append(error[j, j2])
        self._counter += error.shape[0]

        self._labels.append(label[:, -self.detector.offset].cpu())

    def on_test_end(self) -> None:
        # Code derived from TCNPredictionAnomalyDetector
        self._errors = [sum(error) / len(error) for error in self._errors]
        self._errors = torch.stack(self._errors, dim=0)
        labels = torch.cat(self._labels, dim=0)

        self._errors -= self.detector.mean
        scores = F.bilinear(self._errors, self._errors, self.detector.precision.unsqueeze(0)).squeeze(-1)
        scores = scores.cpu()
        assert labels.shape == scores.shape

        self._log_results(labels, scores)
        self._plot_anomalies(labels, scores)


MODEL = LitTCNS2SP
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()

