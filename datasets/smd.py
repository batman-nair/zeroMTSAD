from timesead.data import SMDDataset
from timesead.data.transforms import make_pipe_from_dict, make_dataset_split, DatasetSource
from timesead.data.preprocessing import minmax_scaler

import numpy as np
import lightning.pytorch as lp
import torch
import functools


class SMDDataModule(lp.LightningDataModule):
    SMD_NUM_FEATURES = 38
    def __init__(self, data_params: dict, batch_size: int, train_pipeline: dict={}, test_pipeline: dict={}, standardize: str='minmax'):
        super().__init__()
        self.train_server_ids = data_params['train_server_ids']
        self.val_server_ids = data_params['val_server_ids']
        self.test_server_ids = data_params['test_server_ids']
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.data_splits = data_params['data_splits']
        self.batch_size = batch_size
        self.num_features = self.SMD_NUM_FEATURES
        self.standardize = standardize

    def prepare_data(self) -> None:
        SMDDataset(server_id=1, download=True, preprocess=False)
        # Standardization has to be done dynamically as training server_ids can change
        if self.standardize == 'minmax':
            self._calculate_minmax_stats()
            self.standardize_fn = lambda data, **_: minmax_scaler(data, self.stats)

    def _calculate_minmax_stats(self):
        train_datasets = [SMDDataset(server_id=server_id, training=True, standardize=False) for server_id in self.train_server_ids]
        all_data = []
        for train_dataset in train_datasets:
            data, _ = train_dataset.load_data()
            all_data.append(data)
        all_data = np.vstack(all_data)
        min_vals = all_data.min(axis=0)
        max_vals = all_data.max(axis=0)
        self.stats = {'min': min_vals, 'max': max_vals}

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_datasets = [
                SMDDataset(server_id=server_id, training=True, standardize=self.standardize_fn)
                for server_id in self.train_server_ids]
            self.val_datasets = [
                SMDDataset(server_id=server_id, training=True, standardize=self.standardize_fn)
                for server_id in self.val_server_ids]
            transformed_train_data = [
                make_pipe_from_dict(self.train_pipeline, DatasetSource(train_dataset))
                for train_dataset in self.train_datasets]
            transformed_val_data = [
                make_pipe_from_dict(self.train_pipeline, DatasetSource(val_dataset))
                for val_dataset in self.val_datasets]
            self.combined_train_dataset = torch.utils.data.ConcatDataset(transformed_train_data)
            self.combined_val_dataset = torch.utils.data.ConcatDataset(transformed_val_data)
        elif stage == 'test':
            self.test_datasets = [
                SMDDataset(server_id=server_id, training=False, standardize=self.standardize_fn)
                for server_id in self.test_server_ids]
            transfomed_test_data = [
                make_pipe_from_dict(self.test_pipeline, DatasetSource(ds))
                for ds in self.test_datasets]
            self.combined_test_dataset = torch.utils.data.ConcatDataset(transfomed_test_data)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)


DATASET = SMDDataModule
