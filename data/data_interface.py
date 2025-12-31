import inspect
import importlib
import torch
import typing
from lightning.pytorch import LightningDataModule

class DInterface(LightningDataModule):
    def __init__(self, dataset:str, dataset_params:dict, num_workers:int = 8, batch_size:int = 3, dataset_fn:str = "DataSet"):
        super().__init__()
        self.dataset_fn = dataset_fn
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = dataset_params
        self.batch_size = batch_size
        self.load_data_module()

    def setup(self, stage:typing.Literal["fit", "validate", "test"] = "fit", auto_split_dataset:bool = True):
        if stage == "fit":
            if auto_split_dataset:
                self.trainset, self.valset, self.testset = torch.utils.data.random_split(self.instancialize(), [0.8, 0.1, 0.1], torch.Generator().manual_seed(0))
            else:
                self.trainset = self.instancialize()
        
        if stage == "validate":
            self.valset = self.instancialize()

        if stage == "test":
            self.testset = self.instancialize()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

    def load_data_module(self):
        try:
            self.data_module = getattr(importlib.import_module(f".{self.dataset_fn}", package = ".data"), self.dataset)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{self.dataset_fn}.{self.dataset}')

    def instancialize(self, **other_args):
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)