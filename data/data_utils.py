import os
import numpy as np
import SimpleITK as sitk
import torch

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

def numpy_to_torch_tensor(img:np.ndarray, model = None) -> torch.Tensor:
    img_tensor = torch.from_numpy(img).contiguous()
    img_tensor = img_tensor.to(torch.get_default_dtype())
    if model is not None:
        model.eval()
        img_tensor = model(img_tensor.to(model.device)).detach() 
    return img_tensor

def sitk_to_torch_tensor(img:sitk.Image, model = None) -> torch.Tensor:
    img_array = sitk_to_numpy(img)
    img_array = np.moveaxis(img_array, -1, 0) if img.GetNumberOfComponentsPerPixel() == 3 else np.expand_dims(img_array, axis = 0)
    if img_array.shape[0] == 3:
        img_array = numpy_normalize_rgb(img_array)
    return numpy_to_torch_tensor(img_array, model)

def torch_tensor_to_numpy(input_image:torch.Tensor):
    image_numpy = input_image.cpu().float().numpy()  
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

def crop_2d_image_to_list(img: np.ndarray, window_size: int = 512, stride: int = 512, pad_val_func: str = 'max', crop_way: str = "pad") -> list[list[np.ndarray]]:
    crops = []
    pad_val_func_dict = {'min': np.min, 'max': np.max, 'mean': np.mean}
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h < window_size or img_w < window_size:
        if crop_way == 'delete':
             raise ValueError(f"Image size ({img_h}, {img_w}) is smaller than window size {window_size}, cannot crop with mode 'delete'.")
    h_starts = list(range(0, img_h, stride))
    w_starts = list(range(0, img_w, stride))
    last_h_end = h_starts[-1] + window_size
    last_w_end = w_starts[-1] + window_size
    h_pad_needed = max(0, last_h_end - img_h)
    w_pad_needed = max(0, last_w_end - img_w)
    if crop_way == "pad":
        if h_pad_needed > 0 or w_pad_needed > 0:
            pad_width = [(0, h_pad_needed), (0, w_pad_needed)]
            if img.ndim == 3:
                pad_width.append((0, 0)) 
            pad_val = pad_val_func_dict[pad_val_func](img)
            img = np.pad(img, pad_width, 'constant', constant_values=pad_val)
            img_h, img_w = img.shape[0], img.shape[1]
    elif crop_way == "delete":
        h_starts = [h for h in h_starts if h + window_size <= img_h]
        w_starts = [w for w in w_starts if w + window_size <= img_w]
        if not h_starts or not w_starts:
            return []
    else:
        raise ValueError("crop_way must be 'pad' or 'delete'")
    for h_s in h_starts:
        row_crops = []
        for w_s in w_starts:
            patch = img[h_s : h_s + window_size, w_s : w_s + window_size]
            row_crops.append(patch)
        crops.append(row_crops)
    return crops

def combine_2d_image_from_list(crops:list[list[np.ndarray]]) -> np.ndarray:
    rows = []
    for row_crops in crops:
        row_img = np.concatenate(row_crops, axis = 1)
        rows.append(row_img)
    full_img_np = np.concatenate(rows, axis = 0)
    return full_img_np

def sitk_crop_and_save(input_file:str, output_dir:str, base_name:str = "patch", window_size:int = 512, pad_val_func:str = 'max', save_ext:str = ".png") -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sitk_img = sitk.ReadImage(input_file)
    img_np = sitk_to_numpy(sitk_img)
    crops_list = crop_2d_image_to_list(img_np, window_size, pad_val_func, crop_way="pad")
    for r, row_crops in enumerate(crops_list):
        for c, patch_np in enumerate(row_crops):
            is_vector = True if patch_np.ndim == 3 else False
            save_name = f"{r}_{c}_{base_name}{save_ext}"
            save_path = os.path.join(output_dir, save_name)
            numpy_to_save_img(patch_np.astype(np.uint8), save_path, isVector = is_vector)

def sitk_reconstruct_from_dir(input_dir:str, output_file:str, base_name:str = "patch", ext:str = ".png", model = None) -> None:
    files = [f for f in os.listdir(input_dir) if f.startswith(base_name) and f.endswith(ext)]
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir}")
    patch_map = {}
    max_r, max_c = 0, 0
    for fname in files:
        name_no_ext = os.path.splitext(fname)[0]
        parts = name_no_ext.split('_')
        try:
            row, col = int(parts[-2]), int(parts[-1])
            max_r = max(max_r, row)
            max_c = max(max_c, col)
            img_path = os.path.join(input_dir, fname)
            sitk_patch = sitk.ReadImage(img_path)
            if model is not None:
                model.eval()
                np_patch = torch_tensor_to_numpy(sitk_to_torch_tensor(sitk_patch, model))  
            else:             
                np_patch = sitk_to_numpy(sitk_patch)
            patch_map[(row, col)] = np_patch
        except ValueError:
            print(f"Skipping invalid file: {fname}")
    crops_grid = []
    for r in range(max_r + 1):
        row_list = []
        for c in range(max_c + 1):
            if (r, c) not in patch_map:
                raise ValueError(f"Missing patch: row {r}, col {c}")
            row_list.append(patch_map[(r, c)])
        crops_grid.append(row_list)
    full_img_np = combine_2d_image_from_list(crops_grid)
    is_vector = True if full_img_np.ndim == 3 else False
    numpy_to_save_img(full_img_np.astype(np.uint8), output_file, isVector = is_vector)

def get_gaussian_window(window_size: int, sigma_scale: float = 0.125) -> np.ndarray:
    sigma = window_size * sigma_scale
    x = np.arange(window_size) - window_size // 2
    gaussian_1d = np.exp(-x**2 / (2 * sigma**2))
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    return gaussian_2d / gaussian_2d.max()

def combine_2d_image_with_overlap(crops: list[list[np.ndarray]], stride:int, original_size:list[int, int] = None, mode:str = 'gaussian') -> np.ndarray:
    rows = len(crops)
    cols = len(crops[0])
    patch_h, patch_w = crops[0][0].shape[:2]
    channels = crops[0][0].shape[2] if crops[0][0].ndim == 3 else None
    canvas_h = (rows - 1) * stride + patch_h
    canvas_w = (cols - 1) * stride + patch_w
    if channels:
        canvas = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
        weight_map = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
    else:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    if mode == 'gaussian':
        weight_patch = get_gaussian_window(patch_h)
        if channels:
            weight_patch = weight_patch[..., np.newaxis] 
    else:
        weight_patch = np.ones((patch_h, patch_w), dtype=np.float32)
        if channels:
            weight_patch = weight_patch[..., np.newaxis]
    for i in range(rows):
        for j in range(cols):
            patch = crops[i][j].astype(np.float32)
            top = i * stride
            left = j * stride
            canvas[top : top+patch_h, left : left+patch_w] += (patch * weight_patch)
            weight_map[top : top+patch_h, left : left+patch_w] += weight_patch
    reconstructed_img = canvas / (weight_map + 1e-6)
    reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
    if original_size:
        orig_h, orig_w = original_size
        reconstructed_img = reconstructed_img[:orig_h, :orig_w]
    return reconstructed_img

@DeprecationWarning
def old_crop_2d_image_to_list(img:np.ndarray, window_size:int = 512, pad_val_func:str = 'max', crop_way:str = "pad") -> list[list[np.ndarray]]:
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