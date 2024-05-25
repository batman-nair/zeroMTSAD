from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.reconstruction import AnomalyTransformer, AnomTransf_Loss, AnomTransf_AnomalyDetector
from timesead_experiments.reconstruction.train_anomtransf import get_training_pipeline, get_test_pipeline


class LitAnomalyTransformer(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params):
        super().__init__()
        self.model = AnomalyTransformer(seq_len, num_features, **model_params)
        self.loss = AnomTransf_Loss(lamb=3.0)
        self.save_hyperparameters(model_params)


    def training_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        predictions = self.model(b_inputs[0])
        loss1, loss2, recon_loss = self.loss(predictions, b_targets)
        self.log('train_loss1', loss1, on_epoch=True)
        self.log('train_loss2', loss2, on_epoch=True)
        self.log('train_recon_loss', recon_loss, on_epoch=True)
        self.log('train_loss', loss1 + loss2, on_epoch=True)
        return loss1 + loss2

    def validation_step(self, batch, batch_idx):
        b_inputs, b_targets = batch
        outputs = self.model(b_inputs[0])
        _, loss2, recon_loss = self.loss(outputs, b_targets)
        self.log('val_loss1', recon_loss, on_epoch=True)
        self.log('val_loss2', loss2-recon_loss, on_epoch=True)
        self.log('val_loss', recon_loss, on_epoch=True)
        return recon_loss

    def on_test_start(self) -> None:
        self.detector = AnomTransf_AnomalyDetector(self.model)


MODEL = LitAnomalyTransformer
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
