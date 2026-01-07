import numpy as np
import torch

def convolve2d_cpu(image:np.ndarray, kernel:np.ndarray, out:np.ndarray) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    out_height, out_width = out.shape
    for i in range(out_height):
        for j in range(out_width):
            region = image[i:i + kernel_height, j:j + kernel_width]
            out[i, j] = np.sum(region * kernel)
    return out

def image_convolve2d_cpu(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode = 'constant', constant_values = 0)
    convolved_image = np.zeros_like(image)
    convolve2d_cpu(padded_image, kernel, convolved_image, image.shape[0], image.shape[1], kernel_height, kernel_width)
    return convolved_image

def image_convolve2d_gpu(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    image_tensor = torch.from_numpy(image.astype(np.float32)).to("cuda:0")
    kernel_tensor = torch.from_numpy(kernel.astype(np.float32)).to("cuda:0")
    convolved_image = torch.nn.functional.conv2d(image_tensor.unsqueeze(0).unsqueeze(0), kernel_tensor.unsqueeze(0).unsqueeze(0), padding='same')
    return convolved_image.cpu().numpy().squeeze()

def interpolation_2d_kernel_cpu(image:np.ndarray, blank_coords:np.ndarray, blank_mask:np.ndarray, search_radius:np.uint8, method_index:np.uint8):
    result_array = image.copy()
    for blank_coord in blank_coords:
        i, j = blank_coord
        y_min = max(0, i - search_radius)
        y_max = min(image.shape[0], i + search_radius + 1)
        x_min = max(0, j - search_radius)
        x_max = min(image.shape[1], j + search_radius + 1)
        neighborhood = image[y_min:y_max, x_min:x_max]
        neighborhood_val_mask = ~blank_mask[y_min:y_max, x_min:x_max]
        valid_values = neighborhood[neighborhood_val_mask]
        weights = None
        if method_index == 0:  # mean
            result_array[i, j] = np.mean(valid_values)
        elif method_index == 1:  # median
            result_array[i, j] = np.median(valid_values)
        elif method_index == 2:  # nearest 
            distances = np.sqrt((np.arange(y_min, y_max)[:, None] - i)**2 + (np.arange(x_min, x_max)[None, :] - j)**2)
            valid_distances = distances[neighborhood_val_mask]
            min_dist = np.min(valid_distances)
            nearest_values = valid_values[valid_distances == min_dist]
            result_array[i, j] = np.mean(nearest_values)
        elif method_index == 3:  # linear
            distances = np.sqrt((np.arange(y_min, y_max)[:, None] - i)**2 + (np.arange(x_min, x_max)[None, :] - j)**2)
            valid_distances = distances[neighborhood_val_mask]
            epsilon = 1e-10
            weights = 1.0 / (valid_distances + epsilon)
            weights = weights / weights.sum()  
            result_array[i, j] = np.sum(valid_values * weights)
        elif method_index == 4:  # mode
            bins = 10
            if len(np.unique(valid_values)) > bins:
                hist, bin_edges = np.histogram(valid_values, bins=bins)
                mode_bin = np.argmax(hist)
                bin_mask = (valid_values >= bin_edges[mode_bin]) & \
                           (valid_values < bin_edges[mode_bin + 1])
                mode_values = valid_values[bin_mask]
                result_array[i, j] = np.mean(mode_values) if mode_values.size > 0 else np.mean(valid_values)    
        elif method_index == 5:  # gaussian 
            if weights is None:
                sigma = 1.0
                distances = np.sqrt((np.arange(y_min, y_max)[:, None] - i)**2 + (np.arange(x_min, x_max)[None, :] - j)**2)
                valid_distances = distances[neighborhood_val_mask]
                weights = np.exp(-valid_distances**2 / (2 * sigma**2))
                weights = weights / weights.sum()
            result_array[i, j] = np.sum(valid_values * weights)
        else:
            pass  
    return result_array

def interpolation_2d_kernel_cpu_warpper(image_array:np.ndarray, blank_mask:np.ndarray, search_radius:int = 10, method:int = 5) -> np.ndarray:
    if not search_radius:
        raise("search_radius must be larger than 0")
    if search_radius > min(image_array.shape[0], image_array.shape[1]) // 2:
        raise ValueError("search_radius is too large for the image size")
    if not np.any(blank_mask) or np.all(blank_mask):
        raise ValueError("the blank_mask is all False or all True")
    image_array = np.ascontiguousarray(image_array)
    gpu_mode = torch.cuda.is_available()
    kernel_size = 2 * search_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype = np.uint8)
    convolve2d_func = image_convolve2d_gpu if gpu_mode else image_convolve2d_cpu
    density_map = convolve2d_func(image_array, kernel).astype(np.uint8)
    blank_coords = np.column_stack(np.where(blank_mask))
    densitys = density_map[blank_mask]
    sorted_indices = np.argsort(densitys)[::-1]
    blank_coords = blank_coords[sorted_indices]
    result_array = interpolation_2d_kernel_cpu(image_array, blank_coords, blank_mask, search_radius, method)
    return result_array

def tensor_gaussian_kernel(kernel_size, sigma = 1.0, channels = 1, device='cuda:0'):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def interpolation_2d_kernel_gpu_warpper(image_array:np.ndarray, blank_mask:np.ndarray, blank_coords:np.ndarray, search_radius:int = 10, device:str = "cuda:0", gaussian_params:dict = None) -> np.ndarray:
    image_tensor = torch.from_numpy(image_array.astype(np.float32)).to(device)
    val_mask_tensor = torch.from_numpy(~blank_mask).to(device)
    image_tensor = val_mask_tensor * image_tensor
    image_tensor_pad = torch.nn.functional.pad(image_tensor.unsqueeze(0).unsqueeze(0), (search_radius, search_radius, search_radius, search_radius), mode='constant', value=0).squeeze(0).squeeze(0)
    kernel_size = 2 * search_radius + 1
    gaussian_kernel = tensor_gaussian_kernel(kernel_size, **gaussian_params)
    convolved_image = torch.nn.functional.conv2d(image_tensor_pad.unsqueeze(0).unsqueeze(0), gaussian_kernel, padding='valid').squeeze(0).squeeze(0)
    return convolved_image

if __name__ == "__main__":
    import data_utils
    import SimpleITK as sitk

    raw_img = sitk.ReadImage("/home/t207/Lab_Data_preproc2/allen_data/code/Translation-Different-ISH-Slice/dataset/demo/71112015_raw.jpg")
    expr_img = sitk.ReadImage("/home/t207/Lab_Data_preproc2/allen_data/code/Translation-Different-ISH-Slice/dataset/demo/71112015_expr.jpg")
    traw_np = data_utils.sitk_to_numpy(raw_img)
    texpr_np = data_utils.sitk_to_numpy(expr_img)
    pl_img = traw_np.copy()
    for cl in range(3):
        pl_img[:,:,cl] = interpolation_2d_kernel_cpu_warpper(traw_np[:,:,cl], texpr_np[:,:,cl] > 0, 20, 5) 
    sitk.WriteImage(sitk.GetImageFromArray(pl_img.astype(np.uint8), isVector=True), f"./71112015_raw_interpolated_cpu_{20}.png")