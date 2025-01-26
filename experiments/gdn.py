from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.prediction import GDN, TCNPredictionAnomalyDetector
from timesead_experiments.prediction.train_gdn import get_training_pipeline, get_test_pipeline, get_batch_dim

import torch

def _batched_dot(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    # From timesead.utils.torch_utils
    return torch.matmul(vec1.unsqueeze(-2), vec2.unsqueeze(-1)).squeeze(-1).squeeze(-1)


class LitGDN(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        self.model = GDN(num_features, seq_len, **model_params)
        self.save_hyperparameters(model_params)


    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = TCNPredictionAnomalyDetector(self.model, **detector_params)

    def on_test_start(self) -> None:
        super().on_test_start()
        # Implementing detector code here are as it works over the whole dataset only
        self._errors = []
        self._labels = []
        self._counter = 0

    def test_step(self, batch, batch_idx):
        # Code derived from TCNPredictionAnomalyDetector
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
        self._counter += error.shape[0]

        if batch_idx == 0:
            # Append the first few labels as well
            self._labels.append(label[0, :-1].cpu())
        self._labels.append(label[:, -1].cpu())

    def on_test_end(self) -> None:
        # Code derived from TCNPredictionAnomalyDetector
        self._errors = [sum(error) / len(error) for error in self._errors]
        self._errors = torch.stack(self._errors, dim=0)
        labels = torch.cat(self._labels, dim=0)

        # Compute squared error
        scores = _batched_dot(self._errors, self._errors)
        scores = scores.cpu()

        self._log_results(labels, scores)
        self._plot_anomalies(labels, scores)


MODEL = LitGDN
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
BATCH_DIM = get_batch_dim()
