import numpy as np
import SimpleITK as sitk
import data_utils
import interpolation
import os
'''
    only for 2D-ISH image of allen data processing
'''

DEMO_SECID = 71249739
IMG_BASIC_DIR = "/home/t207/lab_data_preproc4/allen_data/img_data"
DEMO_DATA_DIR = f"./dataset/{DEMO_SECID}"
DEMO_SECIDS = []
DEMO_DONOR_DICT = {
    "6219" : {'Ism1': 71249741, 'Krt222': 71249742, 'Brinp2': 71249743, 'Mical2': 71249740, 'Rassf8': 71249739, 'Ppp4r4': 71249744}
}

def sitk_rgb_to_3channel_gray(image: str | sitk.Image | np.ndarray, weights=(0.299, 0.587, 0.114)) -> np.ndarray:
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    elif isinstance(image, np.ndarray):
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image is not a RGB image!")
        image_np = image.copy()
    else:
        if image.GetNumberOfComponentsPerPixel() != 3:
            raise ValueError("Input image is not a RGB image!")
        image_np = data_utils.sitk_to_numpy(image)
    color_arr = np.array(image_np, dtype=np.float32) / 255.0 
    r, g, b = color_arr[:, :, 0], color_arr[:, :, 1], color_arr[:, :, 2]
    gray_single = weights[0] * r + weights[1] * g + weights[2] * b
    gray_3channel = np.stack([gray_single, gray_single, gray_single], axis=-1)
    gray_3channel = (gray_3channel * 255).astype(np.uint8)
    return gray_3channel

def sitk_compute_otsu_threshold(image:sitk.Image, is_mid:bool = False) -> int:
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetReturnBinMidpoint(is_mid)
    otsu_filter.Execute(image)
    threshold = int(otsu_filter.GetThreshold())
    return threshold

def simple_synthesis_remove_expr_UVstyle_images(raw_image:sitk.Image, expr_image:sitk.Image, otsu_is_mid:bool = False, is_grey:bool = True) -> sitk.Image:
    raw_image_np = data_utils.sitk_to_numpy(raw_image)
    expr_image_np = data_utils.sitk_to_numpy(expr_image)
    if raw_image_np.shape != expr_image_np.shape:
        raise("the shape of raw_image and expr_image must be the same")
    UVs_image_np = np.copy(raw_image_np)
    for cl in range(expr_image_np.ndim):
        expr_image_cl = expr_image_np[:, :, cl]
        UVs_image_np_cl = UVs_image_np[:, :, cl]
        UVs_image_np_cl[expr_image_cl > 0] = sitk_compute_otsu_threshold(sitk.VectorIndexSelectionCast(raw_image, cl), is_mid = otsu_is_mid)
    UVs_image = sitk.GetImageFromArray(UVs_image_np.copy(), isVector = True)
    if is_grey:
        return sitk_rgb_to_3channel_gray(UVs_image) 
    return UVs_image

def make_demo_dataset(win_size:int = 1024, search_radius:int = 20, method:int = 0):
    os.makedirs(f"{DEMO_DATA_DIR}/trainA", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/trainB", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/trainC", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/trainD", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/trainE", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/pl_img", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/rl_img", exist_ok = True)
    os.makedirs(f"{DEMO_DATA_DIR}/xl_img", exist_ok = True)
    expr_img_list= data_utils.make_dataset(f"{IMG_BASIC_DIR}/{DEMO_SECID}/expression")
    raw_img_list= data_utils.make_dataset(f"{IMG_BASIC_DIR}/{DEMO_SECID}/raw")
    print(len(raw_img_list))
    for ix in range(len(raw_img_list)):
        img_name= os.path.split(raw_img_list[ix])[1].split(".")[0]
        print(f"{ix}th: {img_name}")
        img_numpy_crops_list = data_utils.crop_2d_image_to_list(data_utils.sitk_to_numpy(sitk.ReadImage(raw_img_list[ix])), win_size) 
        expr_numpy_crops_list = data_utils.crop_2d_image_to_list(data_utils.sitk_to_numpy(sitk.ReadImage(expr_img_list[ix])), win_size, 'min')
        fake_numpy_crops_list = []
        for w in range(len(img_numpy_crops_list)):
            fake_numpy_crops_list.insert(w, [])
            for h in range(len(img_numpy_crops_list[w])):
                pl_img = img_numpy_crops_list[w][h].copy()
                if np.max(expr_numpy_crops_list[w][h]) > 0:
                    for cl in range(3):
                        pl_img[:,:,cl] = interpolation.interpolation_2d_kernel_cpu_warpper(img_numpy_crops_list[w][h][:,:,cl], expr_numpy_crops_list[w][h][:,:,cl] > 0, search_radius, method) 
                fake_img = sitk_rgb_to_3channel_gray(pl_img)
                real_img = img_numpy_crops_list[w][h]
                expr_img = expr_numpy_crops_list[w][h]
                fake_numpy_crops_list[w].append(fake_img)
                if np.max(expr_numpy_crops_list[w][h]) > 0:
                    data_utils.numpy_to_save_img(pl_img.astype(np.uint8), f"{DEMO_DATA_DIR}/pl_img/{w}_{h}_{img_name}.png", isVector = True)
                    data_utils.numpy_to_save_img(real_img.astype(np.uint8), f"{DEMO_DATA_DIR}/rl_img/{w}_{h}_{img_name}.png", isVector = True)
                    data_utils.numpy_to_save_img(sitk_rgb_to_3channel_gray(real_img).astype(np.uint8), f"{DEMO_DATA_DIR}/xl_img/{w}_{h}_{img_name}.png", isVector = True)
                data_utils.numpy_to_save_img(fake_img.astype(np.uint8), f"{DEMO_DATA_DIR}/trainA/{w}_{h}_{img_name}.png", isVector = True)
                data_utils.numpy_to_save_img(real_img.astype(np.uint8), f"{DEMO_DATA_DIR}/trainB/{w}_{h}_{img_name}.png", isVector = True)
                data_utils.numpy_to_save_img(expr_img.astype(np.uint8), f"{DEMO_DATA_DIR}/trainC/{w}_{h}_{img_name}.png", isVector = True)
                data_utils.numpy_to_save_img(pl_img.astype(np.uint8), f"{DEMO_DATA_DIR}/trainD/{w}_{h}_{img_name}.png", isVector = True)
                data_utils.numpy_to_save_img(sitk_rgb_to_3channel_gray(real_img).astype(np.uint8), f"{DEMO_DATA_DIR}/trainE/{w}_{h}_{img_name}.png", isVector = True)
        data_utils.numpy_to_save_img(data_utils.combine_2d_image_from_list(fake_numpy_crops_list), f"{DEMO_DATA_DIR}/{img_name}_crop_fake.png", isVector = True)
        # break
if __name__ == "__main__":
    # print(data_utils.make_dataset(f"{IMG_BASIC_DIR}/{DEMO_SECID}/expression"))
    make_demo_dataset()
    # raw_img = sitk.ReadImage("/home/t207/Lab_Data_preproc2/allen_data/code/Translation-Different-ISH-Slice/dataset/demo/71112015_raw.jpg")
    # expr_img = sitk.ReadImage("/home/t207/Lab_Data_preproc2/allen_data/code/Translation-Different-ISH-Slice/dataset/demo/71112015_expr.jpg")
    # sitk.WriteImage(simple_synthesis_remove_expr_UVstyle_images(raw_img, expr_img, True, False), f"./71112015_raw_interpolated_cpu_otsu.png")