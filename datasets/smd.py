from timesead.data import SMDDataset
from timesead.data.transforms import make_pipe_from_dict, make_dataset_split, DatasetSource

import lightning.pytorch as lp
import torch


class SMDDataModule(lp.LightningDataModule):
    SMD_NUM_FEATURES = 38
    def __init__(self, data_params: dict, train_pipeline: dict={}, test_pipeline: dict={}):
        super().__init__()
        self.train_server_ids = data_params['train_server_ids']
        self.val_server_ids = data_params['val_server_ids']
        self.test_server_ids = data_params['test_server_ids']
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.data_splits = data_params['data_splits']
        self.num_features = self.SMD_NUM_FEATURES

    def prepare_data(self) -> None:
        SMDDataset(server_id=1, download=True, preprocess=True)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_datasets = [SMDDataset(server_id=server_id, training=True) for server_id in self.train_server_ids]
            self.val_datasets = [SMDDataset(server_id=server_id, training=True) for server_id in self.val_server_ids]
            transformed_train_data = [make_pipe_from_dict(self.train_pipeline, DatasetSource(train_dataset)) for train_dataset in self.train_datasets]
            transformed_val_data = [make_pipe_from_dict(self.train_pipeline, DatasetSource(val_dataset)) for val_dataset in self.val_datasets]
            self.combined_train_dataset = torch.utils.data.ConcatDataset(transformed_train_data)
            self.combined_val_dataset = torch.utils.data.ConcatDataset(transformed_val_data)
        elif stage == 'test':
            self.test_datasets = [SMDDataset(server_id=server_id, training=False) for server_id in self.test_server_ids]
            transfomed_test_data = [make_pipe_from_dict(self.test_pipeline, DatasetSource(ds)) for ds in self.test_datasets]
            self.combined_test_dataset = torch.utils.data.ConcatDataset(transfomed_test_data)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_train_dataset, batch_size=128, num_workers=0, shuffle=False)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_val_dataset, batch_size=128, num_workers=0, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_test_dataset, batch_size=128, num_workers=0, shuffle=False)


DATASET = SMDDataModule
