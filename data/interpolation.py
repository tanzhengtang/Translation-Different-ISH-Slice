import numba
import numpy as np
import torch

@numba.njit(parallel = True)
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

def interpolation_2d_kernel_code(image, blank_coords, blank_mask, search_radius, method_index, result_array):
    x = numba.cuda.grid(1)
    if x < blank_coords.shape[0]:
        i, j = blank_coords[x]
        y_min = max(0, i - search_radius)
        y_max = min(image.shape[0], i + search_radius + 1)
        x_min = max(0, j - search_radius)
        x_max = min(image.shape[1], j + search_radius + 1)
        neighborhood = image[y_min:y_max, x_min:x_max]
        neighborhood_val_mask = ~blank_mask[y_min:y_max, x_min:x_max]
        valid_values = neighborhood[neighborhood_val_mask]
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
            sigma = 1.0
            distances = np.sqrt((np.arange(y_min, y_max)[:, None] - i)**2 + (np.arange(x_min, x_max)[None, :] - j)**2)
            valid_distances = distances[neighborhood_val_mask]
            weights = np.exp(-valid_distances**2 / (2 * sigma**2))
            weights = weights / weights.sum()  
            result_array[i, j] = np.sum(valid_values * weights)
        else:
            pass    

def interpolation_2d_kernel_warpper(image_array:np.ndarray, blank_mask:np.ndarray, search_radius:int = 0, method:int = 0) -> np.ndarray:
    if len(image_array.shape) != 2:
        raise ValueError("must be 2d image for radial interpolation")
    if not search_radius:
        raise("search_radius must be larger than 0")
    if search_radius > min(image_array.shape[0], image_array.shape[1]) // 2:
        raise ValueError("search_radius is too large for the image size")
    if not np.any(blank_mask) or np.all(blank_mask):
        raise ValueError("the blank_mask is all False or all True")
    
    gpu_mode = torch.cuda.is_available()
    result_array = image_array.copy()
    kernel_size = 2 * search_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype = np.uint8)
    convolve2d_func = image_convolve2d_gpu if gpu_mode else image_convolve2d_cpu
    density_map = convolve2d_func(image_array, kernel).astype(np.uint8)
    blank_coords = np.column_stack(np.where(blank_mask))
    densitys = density_map[blank_mask]
    sorted_indices = np.argsort(densitys)[::-1]
    blank_coords = blank_coords[sorted_indices]

    if gpu_mode:
        threads_per_block = 256
        blocks_per_grid = (blank_coords.shape[0] + (threads_per_block - 1)) // threads_per_block
        d_image = numba.cuda.to_device(image_array)
        d_blank_coords = numba.cuda.to_device(blank_coords)
        d_blank_mask = numba.cuda.to_device(blank_mask)
        d_result_array = numba.cuda.to_device(result_array)
        interpolation_2d_kernel_code = numba.cuda.jit(interpolation_2d_kernel_code)
        interpolation_2d_kernel_code[blocks_per_grid, threads_per_block](d_image, d_blank_coords, d_blank_mask, search_radius, method, d_result_array)
        result_array = d_result_array.copy_to_host()
    else:  
        interpolation_2d_kernel_code(image_array, blank_coords, blank_mask, search_radius, method, result_array)
        
    return result_array
