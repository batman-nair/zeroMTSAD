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
    def __init__(self, run_params: dict):
        super().__init__()
        self.testing_step_labels = []
        self.testing_step_scores = []
        self.model = None
        self.loss = torch.nn.MSELoss()
        self.evaluator = timesead.evaluation.Evaluator()
        self.metrics = {
            'best_ts_f1_score': self.evaluator.best_ts_f1_score,
            'ts_auprc': self.evaluator.ts_auprc,
            'best_ts_f1_score_classic': self.evaluator.best_ts_f1_score_classic,
            'ts_auprc_unweighted': self.evaluator.ts_auprc_unweighted,
            'best_f1_score': self.evaluator.best_f1_score,
            'auprc': self.evaluator.auprc,
        }
        self.run_params = run_params
        self.detector = None

    def training_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        predictions = self.model(b_inputs)
        loss = self.loss(predictions, b_targets[0])
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        predictions = self.model(b_inputs)
        loss = self.loss(predictions, b_targets[0])
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.run_params['optimizer']['class'](self.parameters(), **self.run_params['optimizer']['args'])
        scheduler = self.run_params['scheduler']['class'](optimizer, **self.run_params['scheduler']['args'])
        return [optimizer], [scheduler]

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        # Setup self.detector. Can use validation data to initialize the detector
        # self.detector = None
        raise NotImplementedError

    def on_test_start(self) -> None:
        if self.detector is None:
            raise RuntimeError('Detector not initialized. Call setup_detector before testing')

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

        results = {}
        for metric_name, metric_fn in self.metrics.items():
            test_score, other_info = metric_fn(labels, scores)
            results[metric_name] = {'score': test_score, 'other_info': other_info}
        self.logger.log_metrics({name: results[name]['score'] for name in results})
        with open(os.path.join(self.logger.log_dir, 'results.json'), 'w') as ff:
            json.dump(results, ff, indent=4)
        print('Testing score:', results)

        fig = plt.figure()
        ax = plt.axes()
        timesead.utils.plot_utils.plot_sequence_against_anomaly(scores.tolist(), labels.tolist(), ax)
        fig.savefig(os.path.join(self.logger.log_dir, 'anomaly_plot.png'))
        self.logger.experiment.add_figure('anomaly_plot', plt.gcf())

        self.testing_step_labels = []
        self.testing_step_scores = []


MODEL = LitTimeSeADModel
TRAIN_PIPELINE = {}
TEST_PIPELINE = {}
