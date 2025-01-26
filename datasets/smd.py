import numpy as np
import lightning.pytorch as lp
import torch
import functools

from timesead.data import SMDDataset
from timesead.data.dataset import collate_fn
from timesead.data.transforms import make_pipe_from_dict, DatasetSource
from timesead.data.preprocessing import minmax_scaler


class SMDDataModule(lp.LightningDataModule):
    SMD_NUM_FEATURES = 38
    def __init__(
            self,
            data_params: dict,
            batch_size: int,
            train_pipeline: dict={},
            test_pipeline: dict={},
            standardize: str='minmax',
            batch_dim: int=0,
    ):
        super().__init__()
        self.train_ids = data_params['train_ids']
        self.val_ids = data_params['val_ids']
        self.test_ids = data_params['test_ids']
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.validation_split = data_params['validation_split']
        self.num_features = self.SMD_NUM_FEATURES
        self.standardize = standardize
        self.dataloader_params = {
            'batch_size': batch_size,
            'num_workers': data_params['num_workers'],
            'persistent_workers': True,
            'shuffle': False,
            'collate_fn': collate_fn(batch_dim)
        }

    def prepare_data(self) -> None:
        SMDDataset(server_id=0, download=True, standardize=False, preprocess=False)

    def _calculate_minmax_stats(self):
        train_datasets = [SMDDataset(server_id=server_id, training=True, standardize=False) for server_id in self.train_ids]
        all_data = []
        for train_dataset in train_datasets:
            data, _ = train_dataset.load_data()
            all_data.append(data)
        all_data = np.vstack(all_data)
        min_vals = all_data.min(axis=0)
        max_vals = all_data.max(axis=0)
        self.stats = {'min': min_vals, 'max': max_vals}

    def setup(self, stage: str) -> None:
        # Standardization has to be done dynamically as training server_ids can change
        if self.standardize == 'minmax':
            self._calculate_minmax_stats()
            self.standardize_fn = functools.partial(minmax_scaler, stats=self.stats)

        if stage == 'fit':
            self.train_datasets = [
                SMDDataset(server_id=server_id, training=True, standardize=self.standardize_fn)
                for server_id in self.train_ids]
            self.val_datasets = [
                SMDDataset(server_id=server_id, training=True, standardize=self.standardize_fn)
                for server_id in self.val_ids]

            split = 1.0
            if set(self.val_ids) & set(self.train_ids):
                split = self.validation_split

            transformed_train_data = [
                make_pipe_from_dict(self.train_pipeline, DatasetSource(train_dataset, axis='time', end=train_dataset.seq_len))
                for train_dataset in self.train_datasets]
            transformed_val_data = [
                make_pipe_from_dict(self.train_pipeline, DatasetSource(val_dataset, axis='time', end=int(split*val_dataset.seq_len)))
                for val_dataset in self.val_datasets]
            self.combined_train_dataset = torch.utils.data.ConcatDataset(transformed_train_data)
            self.combined_val_dataset = torch.utils.data.ConcatDataset(transformed_val_data)
        elif stage == 'test':
            self.test_datasets = [
                SMDDataset(server_id=server_id, training=False, standardize=self.standardize_fn)
                for server_id in self.test_ids]
            transfomed_test_data = [
                make_pipe_from_dict(self.test_pipeline, DatasetSource(test_dataset, axis='time', end=test_dataset.seq_len))
                for test_dataset in self.test_datasets]
            self.combined_test_dataset = torch.utils.data.ConcatDataset(transfomed_test_data)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_train_dataset, **self.dataloader_params)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_train_dataset, **self.dataloader_params)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_test_dataset, **self.dataloader_params)


DATASET = SMDDataModule
