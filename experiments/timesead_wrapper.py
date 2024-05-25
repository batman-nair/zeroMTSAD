import timesead

import torch
import lightning.pytorch as lp
import os
import json
import matplotlib.pyplot as plt


class LitTimeSeADModel(lp.LightningModule):
    '''
    Implements the defaults for TimeSeAD models
    Note: self.detector should be defined in on_test_start
    '''
    def __init__(self):
        super().__init__()
        self.testing_step_labels = []
        self.testing_step_scores = []
        self.model = None
        self.loss = torch.nn.MSELoss()
        self.evaluator = timesead.evaluation.Evaluator()
        self.metric = self.evaluator.best_ts_f1_score

    def training_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        predictions = self.model(b_inputs[0])
        loss = self.loss(predictions, b_targets)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        predictions = self.model(b_inputs[0])
        loss = self.loss(predictions, b_targets)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
        return [optimizer], [scheduler]

    def on_test_start(self) -> None:
        # self.detector = None
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        batch_scores = self.detector.compute_online_anomaly_score(b_inputs)
        batch_labels = self.detector.format_online_targets(b_targets)
        self.testing_step_scores.append(batch_scores.cpu())
        self.testing_step_labels.append(batch_labels.cpu())

    def on_test_end(self) -> None:
        scores = torch.cat(self.testing_step_scores, dim=0)
        labels = torch.cat(self.testing_step_labels, dim=0)
        assert labels.shape == scores.shape

        test_score, other_info = self.metric(labels, scores)
        self.logger.log_metrics({'test_score': test_score, **other_info})
        with open(os.path.join(self.logger.log_dir, 'results.json'), 'w') as ff:
            json.dump({'test_score': test_score, 'other_info': other_info}, ff)
        print('Testing score:', test_score, other_info)

        fig = plt.figure()
        ax = plt.axes()
        timesead.utils.plot_utils.plot_sequence_against_anomaly(scores.tolist(), labels.tolist(), ax)
        fig.savefig(os.path.join(self.logger.log_dir, 'anomaly_plot.png'))

        self.testing_step_labels = []
        self.testing_step_scores = []


MODEL = LitTimeSeADModel
TRAIN_PIPELINE = {}
TEST_PIPELINE = {}
