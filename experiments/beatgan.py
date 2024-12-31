from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.generative import BeatGANModel, BeatGANReconstructionAnomalyDetector, BeatGANDiscriminatorLoss, BeatGANGeneratorLoss
from timesead_experiments.generative.gan.train_beatgan import get_training_pipeline, get_test_pipeline

import torch

class LitBeatGAN(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        model_params['last_kernel_size'] = seq_len // 32
        model_params_copy = model_params.copy()
        model_params_copy.pop('loss_params')
        self.model = BeatGANModel(num_features, **model_params_copy)
        gen_loss = BeatGANGeneratorLoss(**model_params['loss_params'])
        disc_loss = BeatGANDiscriminatorLoss()
        self.losses = [disc_loss, gen_loss]
        self.loss = None  # Can't assign list of losses to self.loss
        self.save_hyperparameters(model_params)


    def training_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        predictions = self.model(b_inputs)
        loss = sum([l(predictions, b_targets) for l in self.losses])
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        outputs = self.model(b_inputs)
        loss = sum([l(outputs, b_targets) for l in self.losses])
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = BeatGANReconstructionAnomalyDetector(self.model)


MODEL = LitBeatGAN
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
