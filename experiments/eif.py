from experiments.timesead_wrapper import LitTimeSeADModel
from timesead.models.baselines.eif import EIFAD
from timesead_experiments.baselines.train_eif import get_training_pipeline, get_test_pipeline, get_batch_dim
from eif import iForest

import torch

class LitEIFAD(LitTimeSeADModel):
    def __init__(self, seq_len, num_features, model_params, run_params, **kwargs):
        super().__init__(run_params, **kwargs)
        if model_params['extension_level'] == -1:
            model_params['extension_level'] = None
        self.model = EIFAD(**model_params)
        self.model.window_size = seq_len if isinstance(seq_len, int) else seq_len[0]
        self.save_hyperparameters(model_params)
        self.data_full = []

    def training_step(self, batch, batch_idx):
        # Merge all batches as batch processing is not possible
        inputs, targets = batch
        data = inputs[0]
        data = data.reshape(data.shape[0], -1)
        self.data_full.append(data)

    def on_train_end(self):
        self.data_full = torch.cat(self.data_full)
        data = self.data_full.cpu().detach().numpy()
        extension_level = self.model.extension_level
        extension_level = extension_level if extension_level is not None else data.shape[1]-1
        self.model.model = iForest(
            data,
            ntrees=self.model.n_trees,
            sample_size=self.model.sample_size,
            ExtensionLevel=extension_level
        )


    def validation_step(self, batch, batch_idx):
        # Log dummy validation loss for early stopping callback
        self.log('val_loss', 0, on_epoch=True)

    def setup_detector(self, detector_params: dict, val_loader: torch.utils.data.DataLoader) -> None:
        self.detector = self.model

    def configure_optimizers(self):
        return None


MODEL = LitEIFAD
TRAIN_PIPELINE = get_training_pipeline()
TEST_PIPELINE = get_test_pipeline()
BATCH_DIM = get_batch_dim()
