from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
import torch
import wandb

class SaveMiddleCallback(Callback):
    def __init__(self, log_frequency: int = 1):
        self.log_frequency = log_frequency

    @staticmethod
    def _wb_logger(trainer) -> WandbLogger | None:
        lg = trainer.logger
        if isinstance(lg, WandbLogger):
            return lg

        if isinstance(lg, (list, tuple)):
            return next((l for l in lg if isinstance(l, WandbLogger)), None)
        return None

    def _slice_images(self, vol: torch.Tensor) -> torch.Tensor:
        D = vol.shape[-1]
        mid_idx = D // 2
        return vol[..., mid_idx]

    def _get_image_pair(self, batch: dict, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch,dict):
            x_0 =  batch.get("image_x_0")
        else:
            x_0 = batch[0].get("image_x_0")
        return x_0, outputs

    def _log_3d_comparison(self, wb: WandbLogger, x_0: torch.Tensor, preds: torch.Tensor) -> None:
        gt_slice = self._slice_images(x_0[0].squeeze(0).cpu())
        pd_slice = self._slice_images(preds[0].squeeze(0).cpu())
        pair = torch.concat([pd_slice,gt_slice],dim=1)
        wb.experiment.log({"val_pair": wandb.Image(pair.numpy(), caption="prediction---------ground_truth")})

    def _log_2d_comparison(self, wb: WandbLogger, x_0: torch.Tensor, preds: torch.Tensor) -> None:
        gt_slice = x_0[0, 0, :, :]
        pd_slice = preds[0, 0, :, :]
        pair = torch.concat([pd_slice, gt_slice], dim=1)
        wb.experiment.log({"val_pair": wandb.Image(pair.cpu().numpy(), caption="prediction---------ground_truth")})

    def on_validation_batch_end(
            self,
            trainer,
            outputs,
            batch,
            batch_idx,
            *_
    ) -> None:
        if batch_idx != 0:
            return

        wb = self._wb_logger(trainer)
        if wb is None:
            return

        x_0, preds = self._get_image_pair(batch, outputs)

        if x_0.ndim == 5:
            self._log_3d_comparison(wb, x_0, preds)
        else:
            self._log_2d_comparison(wb, x_0, preds)