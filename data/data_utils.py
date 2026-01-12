import os
import numpy as np
import SimpleITK as sitk
import torch
import torchvision

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
    '.nrrd', '.nii.gz'
]

def torch_tensor_to_sitk(img:torch.Tensor, spacing:int = 1) -> sitk.Image:
    img = img.detach().cpu().numpy().transpose(1,2,3,0)
    is_v = True
    if img.shape[3] == 1:
        img = img.squeeze(3)
        is_v = False
    res = sitk.GetImageFromArray(img, isVector = is_v)
    res.SetSpacing((spacing,spacing,spacing))
    return res

def numpy_to_save_img(img_np:np.ndarray, img_path:str, **args) -> bool:
    return sitk.WriteImage(sitk.GetImageFromArray(np.ascontiguousarray(img_np), **args), img_path)

def sitk_to_numpy(img:sitk.Image) -> np.ndarray:
    if len(img.GetSize()) == 2:
        pass
    elif len(img.GetSize()) == 3:
        sitk.PermuteAxes(img, [2,1,0])
    else:
        raise("error image is 1d image or higher than 3d image")
    return sitk.GetArrayFromImage(img)

def numpy_normalize_rgb(img:np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    return (img_float - 127.5) / 127.5

def numpy_denormalize_rgb(img_norm:np.ndarray):
    if img_norm.max() > 1 or img_norm.min() < -1:
        raise ValueError('It seems that the img is not normalized since its max value > 1 or min value < -1!')
    img_denorm = img_norm * 127.5 + 127.5
    img_denorm = np.clip(img_denorm, 0, 255)
    img_uint8 = img_denorm.astype(np.uint8)
    return img_uint8

def sitk_to_torch_tensor(img:sitk.Image) -> torch.Tensor:
    img_array = sitk_to_numpy(img)
    img_array = np.moveaxis(img_array, -1, 0) if img.GetNumberOfComponentsPerPixel() == 3 else np.expand_dims(img_array, axis = 0)
    if img_array.shape[0] == 3:
        img_array = numpy_normalize_rgb(img_array)
    img_tensor = torch.from_numpy(img_array).contiguous()
    img_tensor = img_tensor.to(torch.get_default_dtype())
    # print(img_t.min(), img_t.max()) 
    # std_mean = (0.5,) if img.GetNumberOfComponentsPerPixel() == 1 else (0.5, 0.5, 0.5)
    # img_t = torchvision.transforms.Normalize(std_mean, std_mean)(img_t) if is_normalize else img_t
    # print(img_t.min(), img_t.max()) 
    return img_tensor

def torch_tensor_to_numpy(input_image:torch.Tensor):
    image_numpy = input_image.squeeze(0).cpu().float().numpy()  # convert it into a numpy array
    image_numpy = np.moveaxis(image_numpy, 0, -1)
    if image_numpy.shape[-1] == 3:
        image_numpy = numpy_denormalize_rgb(image_numpy)
    return image_numpy

def resample_image_specific_spacing(image:sitk.Image, new_spacing:list) -> sitk.Image:
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    if image.GetDimension() == 2:
        new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                    int(round(original_size[1] * (original_spacing[1] / new_spacing[1])))]
    elif image.GetDimension() == 3:
        new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                    int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                    int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    else:
        print(f"image.GetDimension() is equal to {image.GetDimension()}")
        return 0
    return sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, image.GetOrigin(), new_spacing, image.GetDirection(), 0.0, outputPixelType = image.GetPixelIDValue())

def sitk_downsample_write_file(image_path:str, scale_factor:int = 2) -> str:
    downsample_image_path = os.path.join(os.path.split(image_path)[0], f"ds{scale_factor}_" + os.path.split(image_path)[1])
    sitk.WriteImage(sitk_downsample(sitk.ReadImage(image_path), scale_factor), downsample_image_path)
    return downsample_image_path

def sitk_downsample(image:sitk.Image, scale_factor:int = 2):
    new_spacing = [sp * scale_factor for sp in image.GetSpacing()]
    return resample_image_specific_spacing(image, new_spacing)

def sitk_upsample(image:sitk.Image, scale_factor:int = 2):
    new_spacing = [sp // scale_factor for sp in image.GetSpacing()]
    return resample_image_specific_spacing(image, new_spacing)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def fix_tensor_shape(input_tensor:torch.Tensor, shape_divisor:int = 4) -> torch.Tensor:
    if len(input_tensor.shape) != 5:
        raise("the input tenosr must be (batch, channel, d, h ,w)")
    pad_list = []
    for shape_num in input_tensor.shape[2:5]:
        rn = shape_num % shape_divisor
        if rn % 2:
            pad_list.append(int(rn // 2) + 1)
            pad_list.append(int(rn // 2))
        else:
            pad_list.append(int(rn / 2))
            pad_list.append(int(rn / 2))
    return torch.nn.functional.pad(input_tensor, pad_list)

def crop_2d_image_to_list(img:np.ndarray, window_size:int = 512, pad_val_func:str = 'max', crop_way:str = "pad") -> list[list[np.ndarray]]:
    '''
        Crop the 2D image into patches with given crop_size.
        Normally the crops is a row-col major order which contains all crops in the first row, then the second row, etc.
        All the crop images are with the size of (window_size, window_size). Last crops in each row/column are padded or deleted if the image size is not divisible by window_size. 
    '''
    crops = []
    pad_val_func_dict = {'min': np.min, 'max': np.max, 'mean': np.mean}
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h < window_size or img_w < window_size:
        raise(f"the image size ({img_h}, {img_w}) is smaller than crop size {window_size}")
    h_pad = img_h % window_size
    w_pad = img_w % window_size
    pan_model = [(0, window_size - h_pad if h_pad !=0 else 0), (0, window_size - w_pad if w_pad !=0 else 0)]
    if img.ndim == 3:
        pan_model.append((0,0))
    if crop_way == "pad":
        re_img = np.pad(img, pan_model, 'constant', constant_values = pad_val_func_dict[pad_val_func](img))
    elif crop_way == "delete":
        re_img = img[0:img_h - h_pad, 0:img_w - w_pad] if img.ndim == 2 else img[0:img_h - h_pad, 0:img_w - w_pad, :]
    else:
        raise("crop_way must be pad or delete")
    re_h, re_w = re_img.shape[0], re_img.shape[1]
    h_index_range = re_h // window_size 
    w_index_range = re_w // window_size 
    for hs in range(h_index_range):
        hs_list = []
        for ws in range(w_index_range):
            crop_img = re_img[hs*window_size:(hs+1)*window_size, ws*window_size:(ws+1)*window_size] if img.ndim == 2 else re_img[hs*window_size:(hs+1)*window_size, ws*window_size:(ws+1)*window_size, :]
            hs_list.append(crop_img)
        crops.append(hs_list)
    return crops

def combine_2d_image_from_list(crops:list[list[np.ndarray]]) -> np.ndarray:
    '''
        Combine the crops into a full image.
        The input crops is a row-col major order which contains all crops in the first row, then the second row, etc.
    '''
    rows = []
    for row_crops in crops:
        row_img = np.concatenate(row_crops, axis = 1)
        rows.append(row_img)
    full_img_np = np.concatenate(rows, axis = 0)
    return full_img_np