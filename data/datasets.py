import torch
import numpy as np
import SimpleITK as sitk
from data import data_utils

# TODO. add the data augmentation func
class CommonDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, mA_name:str, mB_name:str, downsample_factor:int = 0):
        super().__init__()
        self.mA_list = sorted(data_utils.make_dataset(f"{dataset_path}/{mA_name}"))
        self.mB_list =  sorted(data_utils.make_dataset(f"{dataset_path}/{mB_name}"))
        self.downsample_factor = downsample_factor
    
    def _get_image_index(self, index):
        return index, index

    def __getitem__(self, index):
        index_A, index_B = self._get_image_index(index)
        A_file_path = self.mA_list[index_A]
        B_file_path = self.mB_list[index_B]
        A_img = sitk.ReadImage(A_file_path)
        B_img = sitk.ReadImage(B_file_path)
        if self.downsample_factor >= 2:
            A_img = data_utils.sitk_downsample(A_img, self.downsample_factor)
            B_img = data_utils.sitk_downsample(B_img, self.downsample_factor)
        return data_utils.sitk_to_torch_tensor(A_img), data_utils.sitk_to_torch_tensor(B_img)
    
    def __len__(self):
        return np.max([len(self.mA_list), len(self.mB_list)])

class UnalignedDataSet(CommonDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_image_index(self, index):
        return index, np.random.randint(0, index)

class AlignedDataSet(CommonDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# TODO. to inplement the code of patch dataset.
class PatchDataset(CommonDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

DATASETS_CLASS_DICT = dict(AlignedDataSet = AlignedDataSet,
                UnalignedDataSet = UnalignedDataSet)