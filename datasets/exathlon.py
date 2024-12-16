import numpy as np
import lightning.pytorch as lp
import torch
import functools

from timesead.data import ExathlonDataset
from timesead.data.transforms import make_pipe_from_dict, DatasetSource
from timesead.data.preprocessing import minmax_scaler


class ExathlonDataModule(lp.LightningDataModule):
    NUM_FEATURES = 19
    def __init__(self, data_params: dict, batch_size: int, train_pipeline: dict={}, test_pipeline: dict={}, standardize: str='minmax'):
        super().__init__()
        self.train_ids = data_params['train_ids']
        self.val_ids = data_params['val_ids']
        self.test_ids = data_params['test_ids']
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.validation_split = data_params['validation_split']
        self.batch_size = batch_size
        self.num_features = self.NUM_FEATURES
        self.standardize = standardize
        self.num_workers = data_params['num_workers']

    def prepare_data(self) -> None:
        ExathlonDataset(app_id=1, download=True, standardize=False, preprocess=True)

    def _calculate_minmax_stats(self):
        train_datasets = [ExathlonDataset(app_id=app_id, training=True, standardize=False) for app_id in self.train_ids]
        all_data = []
        for train_dataset in train_datasets:
            for index in range(len(train_dataset)):
                data, _ = train_dataset[index]
                assert len(data) == 1
                all_data.append(data[0])
        all_data = np.vstack(all_data)
        min_vals = all_data.min(axis=0)
        max_vals = all_data.max(axis=0)
        self.stats = {'min': min_vals, 'max': max_vals}

    def setup(self, stage: str) -> None:
        # Standardization has to be done dynamically as training app_ids can change
        if self.standardize == 'minmax':
            self._calculate_minmax_stats()
            self.standardize_fn = functools.partial(minmax_scaler, stats=self.stats)
        if stage == 'fit':
            self.train_datasets = [
                ExathlonDataset(app_id=app_id, training=True, standardize=self.standardize_fn)
                for app_id in self.train_ids]
            self.val_datasets = [
                ExathlonDataset(app_id=app_id, training=True, standardize=self.standardize_fn)
                for app_id in self.val_ids]

            split = 1.0
            if set(self.val_ids) & set(self.train_ids):
                split = self.validation_split

            transformed_train_data = [
                make_pipe_from_dict(self.train_pipeline, DatasetSource(train_dataset, axis='time', end=train_dataset.seq_len))
                for train_dataset in self.train_datasets]
            transformed_val_data = [
                make_pipe_from_dict(self.train_pipeline, DatasetSource(val_dataset, axis='time', end=[int(split*seq_len) for seq_len in val_dataset.seq_len]))
                for val_dataset in self.val_datasets]
            self.combined_train_dataset = torch.utils.data.ConcatDataset(transformed_train_data)
            self.combined_val_dataset = torch.utils.data.ConcatDataset(transformed_val_data)
        elif stage == 'test':
            self.test_datasets = [
                ExathlonDataset(app_id=app_id, training=False, standardize=self.standardize_fn)
                for app_id in self.test_ids]
            transfomed_test_data = [
                make_pipe_from_dict(self.test_pipeline, DatasetSource(test_dataset, axis='time', end=test_dataset.seq_len))
                for test_dataset in self.test_datasets]
            self.combined_test_dataset = torch.utils.data.ConcatDataset(transfomed_test_data)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.combined_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


DATASET = ExathlonDataModule
