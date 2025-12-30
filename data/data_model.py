import torch
import numpy as np
import SimpleITK as sitk
from data import data_utils

class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, mA_name:str, mB_name:str, downsample_factor:int = 0):
        super().__init__()
        self.mA_list = sorted(data_utils.make_dataset(f"{dataset_path}/{mA_name}"))
        self.mB_list =  sorted(data_utils.make_dataset(f"{dataset_path}/{mB_name}"))
        self.downsample_factor = downsample_factor

    def __getitem__(self, index):
        A_index = index
        B_index = np.random.randint(0, len(self.mB_list))
        A_file_path = self.mA_list[A_index]
        B_file_path = self.mB_list[B_index]
        A_img = sitk.ReadImage(A_file_path)
        B_img = sitk.ReadImage(B_file_path)
        if self.downsample_factor >= 2:
            A_img = data_utils.sitk_downsample(A_img, self.downsample_factor)
            B_img = data_utils.sitk_downsample(B_img, self.downsample_factor)
        return [data_utils.sitk_to_torch_tensor(A_img), data_utils.sitk_to_torch_tensor(B_img)]

    def __len__(self):
        return len(self.mA_list)

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, mA_name:str, mB_name:str, downsample_factor:int = 0):
        super().__init__()
        self.mA_list = sorted(data_utils.make_dataset(f"{dataset_path}/{mA_name}"))
        self.mB_list =  sorted(data_utils.make_dataset(f"{dataset_path}/{mB_name}"))
        self.downsample_factor = downsample_factor

    def __getitem__(self, index):
        A_index = index
        B_index = index
        A_file_path = self.mA_list[A_index]
        B_file_path = self.mB_list[B_index]
        A_img = sitk.ReadImage(A_file_path)
        B_img = sitk.ReadImage(B_file_path)
        if self.downsample_factor >= 2:
            A_img = data_utils.sitk_downsample(A_img, self.downsample_factor)
            B_img = data_utils.sitk_downsample(B_img, self.downsample_factor)
        return [data_utils.sitk_to_torch_tensor(A_img), data_utils.sitk_to_torch_tensor(B_img)]

    def __len__(self):
        return len(self.mA_list)

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, mA_name:str, mB_name:str, downsample_factor:int = 0):
        super().__init__()
        self.mA_list = sorted(data_utils.make_dataset(f"{dataset_path}/{mA_name}"))
        self.mB_list =  sorted(data_utils.make_dataset(f"{dataset_path}/{mB_name}"))
        self.downsample_factor = downsample_factor