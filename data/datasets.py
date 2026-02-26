import torch
import numpy as np
import SimpleITK as sitk
from data import data_utils
import bisect
from PIL import Image

# TODO. add the data augmentation func
class CommonDataSet(torch.utils.data.Dataset):
    def __init__(self, mA_path:str, mB_path:str, downsample_factor:int = 0, is_shuffle:bool = False):
        super().__init__()
        self.mA_path = mA_path
        self.mB_path = mB_path
        self.downsample_factor = downsample_factor
        self.is_shuffle = is_shuffle
        self._load_imgs_path()
    
    def _load_imgs_path(self):
        self.mA_list = sorted(data_utils.make_dataset(f"{self.mA_path}"))
        self.mB_list = sorted(data_utils.make_dataset(f"{self.mB_path}"))
        
    def _get_image_index(self, index):
        if self.is_shuffle:
            return index, np.random.randint(-1, index)
        else:
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
        return np.min([len(self.mA_list), len(self.mB_list)])

class PredictDataSet(CommonDataSet):
    def __init__(self, **kwargs):
        kwargs['is_shuffle'] = False
        super().__init__(**kwargs)
    
    def __getitem__(self, index):
        index_A, index_B = self._get_image_index(index)
        A_file_path = self.mA_list[index_A]
        B_file_path = self.mB_list[index_B]
        A_img = sitk.ReadImage(A_file_path)
        B_img = sitk.ReadImage(B_file_path)
        if self.downsample_factor >= 2:
            A_img = data_utils.sitk_downsample(A_img, self.downsample_factor)
            B_img = data_utils.sitk_downsample(B_img, self.downsample_factor)
        return data_utils.sitk_to_torch_tensor(A_img), A_file_path

class UnalignedDataSet(CommonDataSet):
    def __init__(self, **kwargs):
        kwargs['is_shuffle'] = True
        super().__init__(**kwargs)

class AlignedDataSet(CommonDataSet):
    def __init__(self, **kwargs):
        kwargs['is_shuffle'] = False
        super().__init__(**kwargs)

class PatchDataset(CommonDataSet):
    def __init__(self, patch_size:int, stride:int, padding_mode:str = "max", **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.mA_cumulative_sizes, self.mA_image_metas = self._load_stride_imgs_path(patch_size, stride)
        self.mB_cumulative_sizes, self.mB_image_metas = self._load_stride_imgs_path(patch_size, stride)    
        
    def _load_stride_imgs_path(self, image_paths:list, stride:int) -> tuple[list, dict]:
        cumulative_sizes = []
        image_meta = []
        total_patches = 0
        for p in image_paths:
            with Image.open(p) as img:
                w, h = img.size
            n_rows = np.ceil(h / stride)
            n_cols = np.ceil(w / stride)
            count = n_rows * n_cols
            image_meta.append({
                "path": p,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "width": w,  
                "height": h,
                "count": count
            })
            total_patches += count
            cumulative_sizes.append(total_patches)
        return cumulative_sizes, image_meta

    def _get_fill_color(self, img_patch:Image.Image, padding_mode:str):
            extrema = img_patch.getextrema()
            if img_patch.mode == 'L' or img_patch.mode == 'I;16':
                if padding_mode == 'min':
                    return extrema[0]
                else:
                    return extrema[1] 
            elif img_patch.mode == 'RGB': 
                if padding_mode == 'min':
                    return tuple(ch[0] for ch in extrema)
                else:
                    return tuple(ch[1] for ch in extrema)
            return 0 if padding_mode == 'min' else 255

    def _process_get_item(self, index:int, cumulative_sizes:list, image_meta_list:dict, patch_size:int):
        image_idx = bisect.bisect_right(cumulative_sizes, index) - 1
        image_meta = image_meta_list[image_idx]
        local_idx = index - cumulative_sizes[image_idx]
        row = local_idx // image_meta['n_cols']
        col = local_idx % image_meta['n_cols']
        y = row * self.stride
        x = col * self.stride
        with Image.open(image_meta['path']) as img:
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            if patch.size != (patch_size, patch_size):
                padding_value = self._get_fill_color(patch, self.padding_mode)
                new_patch = Image.new(img.mode, (patch_size, patch_size), padding_value)
                new_patch.paste(patch, (0, 0))
                patch = new_patch
        patch_array = np.array(img)
        if patch.mode == 'RGB':
            patch_array = np.moveaxis(patch_array, -1, 0)
            patch_array = data_utils.numpy_normalize_rgb(patch_array)
        return data_utils.numpy_to_torch_tensor(patch_array)
    
    def __getitem__(self, index):
        index_A, index_B = self._get_image_index(index)
        return self._process_get_item(index_A, self.mA_cumulative_sizes, self.mA_image_metas, self.patch_size), self._process_get_item(index_B, self.mB_cumulative_sizes, self.mA_image_metas, self.patch_size)

    def __len__(self):
        return np.min(self.mA_cumulative_sizes[-1], self.mB_cumulative_sizes[-1])
    
DATASETS_CLASS_DICT = dict(AlignedDataSet = AlignedDataSet,
                UnalignedDataSet = UnalignedDataSet,
                PredictDataSet = PredictDataSet,
                PatchDataset = PatchDataset)