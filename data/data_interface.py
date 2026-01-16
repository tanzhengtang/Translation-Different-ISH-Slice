import torch
import typing
from lightning.pytorch import LightningDataModule
from data import datasets

class DataInterface(LightningDataModule):
    def __init__(self, dataset:str, dataset_params:dict, num_workers:int = 8, batch_size:int = 3, data_rate:list = [0.7, 0.2, 0.1]):
        super().__init__()
        self.dataset = dataset
        self.dataset_params = dataset_params
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_rate = data_rate
        # self.save_hyperparameters()
        self.load_data_module()
        
    def load_data_module(self):
        if self.dataset not in datasets.DATASETS_CLASS_DICT.keys():
            raise ValueError(f"No such {self.dataset} is implemented")
        self._dataset = datasets.DATASETS_CLASS_DICT[self.dataset](**self.dataset_params)

    def setup(self, stage:typing.Literal["fit", "validate", "test"] = "fit", auto_split_dataset:bool = True):
        if stage == "fit":
            if auto_split_dataset:
                self.trainset, self.valset, self.testset = torch.utils.data.random_split(self._dataset, self.data_rate, torch.Generator().manual_seed(77))
            else:
                self.trainset = self._dataset
        if stage == "validate":
            self.valset = self._dataset
        if stage == "test":
            self.testset = self._dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)