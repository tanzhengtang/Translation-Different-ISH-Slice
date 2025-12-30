import numpy as np
import SimpleITK as sitk
import data_utils
import scipy

'''
    only for 2D-ISH image of allen data processing
'''
def sitk_rgb_to_3channel_gray(image: str | sitk.Image, weights=(0.299, 0.587, 0.114)) -> np.ndarray:
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    if image.GetNumberOfComponentsPerPixel() != 3:
        raise ValueError("Input image is not a RGB image!")
    image_np = data_utils.sitk_to_numpy(image)
    color_arr = np.array(image_np, dtype=np.float32) / 255.0 
    r, g, b = color_arr[:, :, 0], color_arr[:, :, 1], color_arr[:, :, 2]
    gray_single = weights[0] * r + weights[1] * g + weights[2] * b
    gray_3channel = np.stack([gray_single, gray_single, gray_single], axis=-1)
    gray_3channel = (gray_3channel * 255).astype(np.uint8)
    return sitk.GetImageFromArray(gray_3channel, isVector = True)

def sitk_compute_otsu_threshold(image:sitk.Image, is_mid:bool = False) -> int:
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetReturnBinMidpoint(is_mid)
    otsu_filter.Execute(image)
    threshold = int(otsu_filter.GetThreshold())
    return threshold

def synthesis_remove_expr_UVstyle_images(raw_image:sitk.Image, expr_image:sitk.Image, otsu_is_mid:bool = False) -> sitk.Image:
    raw_image_np = data_utils.sitk_to_numpy(raw_image)
    expr_image_np = data_utils.sitk_to_numpy(expr_image)
    if raw_image_np.shape != expr_image_np.shape:
        raise("the shape of raw_image and expr_image must be the same")
    UVs_image_np = np.copy(raw_image_np)
    for cl in range(expr_image_np.ndim):
        expr_image_cl = expr_image_np[:, :, cl]
        UVs_image_np_cl = UVs_image_np[:, :, cl]
        UVs_image_np_cl[expr_image_cl > 0] = sitk_compute_otsu_threshold(sitk.VectorIndexSelectionCast(raw_image, cl), is_mid = otsu_is_mid)
    UVs_image = sitk_rgb_to_3channel_gray(sitk.GetImageFromArray(UVs_image_np.copy(), isVector = True))
    return UVs_image



if __name__ == "__main__":
    raw_img = sitk.ReadImage("/home/t207/Translation-Different-ISH-Slice/dataset/demo/71112015_raw.jpg")
    expr_img = sitk.ReadImage("/home/t207/Translation-Different-ISH-Slice/dataset/demo/71112015_expr.jpg")
    traw_np = data_utils.sitk_to_numpy(raw_img)[:,:,0]
    texpr_np = data_utils.sitk_to_numpy(expr_img)[:,:,0]
    blank_mask = texpr_np > 0
    # result = radial_interpolation_2d_mask_pixel(traw_np, blank_mask, search_radius = 50, method = 'nearest')
    sitk.WriteImage(sitk.GetImageFromArray(result), "/home/t207/Translation-Different-ISH-Slice/dataset/demo/71112015_raw_interpolated.png")