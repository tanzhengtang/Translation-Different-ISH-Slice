from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import BasePredictionWriter
import torch
import wandb
import numpy as np
from data import data_utils
import os

def cgan_check(wb, pl_module, source_img_path, fake_img_path, win_size, stride, discard_val:int = 240, discard_ratio:float = 0.05) -> None:
    import SimpleITK as sitk
    import os 
    img_np = data_utils.sitk_to_numpy(sitk.ReadImage(source_img_path))
    fake_img_dir, fake_img_fn = os.path.split(fake_img_path)
    sitk_np_patchs = data_utils.crop_2d_image_to_list(img_np, win_size, stride)
    tensor_crop_patchs = []
    for rows in sitk_np_patchs:
        np_patchs = []
        for col in rows:
            if col.mean() > discard_val or (col < discard_val).mean() < discard_ratio: 
                np_patchs.append(col)
                continue 
            col = np.moveaxis(col, -1, 0)
            np_patchs.append(data_utils.torch_tensor_to_numpy(data_utils.numpy_to_torch_tensor(col, pl_module)))
        tensor_crop_patchs.append(np_patchs)
    data_utils.numpy_to_save_img(data_utils.combine_2d_image_with_overlap(tensor_crop_patchs, stride, img_np.shape[0:2]), f"{fake_img_dir}/{wb.version}_{pl_module.trainer.current_epoch}_{fake_img_fn}", isVector = True)
    pl_module.train()
        
def p2p_check(wb, pl_module, val_img_dir, output_img_dir, base_name) -> None:
    data_utils.sitk_reconstruct_from_dir(val_img_dir, f"{output_img_dir}/{base_name}_{wb.version}_{pl_module.trainer.current_epoch}.png", str(base_name), ".png", pl_module)
    pl_module.train()

CHECK_FUNC_DICT  = dict(cgan_check = cgan_check, p2p_check = p2p_check)

class SaveMiddleCallback(Callback):
    def __init__(self, log_frequency: int = 1, check_func_name:str = "", check_func_params:dict = {}):
        self.check_func = CHECK_FUNC_DICT[check_func_name] if check_func_name else None
        self.log_frequency = log_frequency
        self.check_func_params = check_func_params

    @staticmethod
    def _wb_logger(trainer) -> WandbLogger | None:
        lg = trainer.logger
        if isinstance(lg, WandbLogger):
            return lg
        if isinstance(lg, (list, tuple)):
            return next((l for l in lg if isinstance(l, WandbLogger)), None)
        return None

    def _log_2d_comparison(self, wb:WandbLogger, input_x:torch.Tensor, input_y:torch.Tensor, preds:torch.Tensor) -> None:
        gs_slice, gt_slice, pd_slice = data_utils.torch_tensor_to_numpy(input_x[0,::,]), data_utils.torch_tensor_to_numpy(input_y[0,::,]), data_utils.torch_tensor_to_numpy(preds[0,::,])
        pair = np.concatenate([pd_slice, gt_slice, gs_slice], axis = 1)
        wb.experiment.log({"val_pair": wandb.Image(pair.astype(np.uint8), caption="prediction---------ground_truth")})
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *_) -> None:
        if batch_idx != 0:
            return
        wb = self._wb_logger(trainer)
        if wb is None:
            return
        input_x, input_y, preds = batch[0], batch[1], outputs
        self._log_2d_comparison(wb,input_x, input_y, preds)
        if self.check_func is not None:
            self.check_func(wb, pl_module, **self.check_func_params)

class OverrideEpochCallback(Callback):
    def __init__(self, start_epoch:int = 0):
        super().__init__()
        self.start_epoch = start_epoch

    def on_train_start(self, trainer, pl_module):
        if self.start_epoch > 0:
            print(f"override Epoch: {self.start_epoch}")
            trainer.fit_loop.epoch_progress.current.completed = self.start_epoch

class ImagePredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        img_np = data_utils.torch_tensor_to_numpy(prediction[0,::,])
        _, fn = os.path.split(batch[1][0])
        save_path = os.path.join(self.output_dir, f"{fn}")
        data_utils.numpy_to_save_img(img_np, save_path, isVector = True)