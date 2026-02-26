import torch
import itertools
from model import networks

class PatchCganInterface(networks.GanCommonModel):
    def __init__(self, direction:str, pool_image_size:int, patch_size = 512, stirde = 512, **kwargs):
        super().__init__(**kwargs)

    def filter_patches(self, patches_A, patches_B, threshold:int = 0.1):
        stds = patches_A.view(patches_A.size(0), -1).std(dim=1)
        valid_indices = stds > threshold
        if valid_indices.sum() == 0:
            return None, None
        return patches_A[valid_indices], patches_B[valid_indices]

    def tensor_to_patches(self, img_tensor):
        b, c, h, w = img_tensor.shape
        patches = img_tensor.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, self.patch_size, self.patch_size)
        return patches

    def load_networks(self):
        self.netG_A = networks.NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.netG_B = networks.NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.netD_A = networks.NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        self.netD_B = networks.NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        pool_image_size = self.hparams.get('pool_image_size', 50)
        self.fake_A_pool = networks.ImagePool(pool_image_size)
        self.fake_B_pool = networks.ImagePool(pool_image_size)
        self.direction = self.hparams.get('direction', 'AtoB')
        if self.direction == 'AtoB':
            self.netG = self.netG_A
        else:
            self.netG = self.netG_B
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionCycle = torch.nn.L1Loss()

    def backward_G(self, batch):
        self.real_A, self.real_B = batch
        self.fake_B = self.netG_A(self.real_A)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)
        self.rec_A = self.netG_B(self.fake_B)

        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0
        if lambda_idt > 0:
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        self.manual_backward(loss_G)
        return loss_G_A, loss_G_B, loss_G

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.manual_backward(loss_D)
        return loss_D

    def backward_D(self):
        if hasattr(self.hparams, 'pool_image_size'):
            fake_B = self.fake_B_pool.query(self.fake_B)
            fake_A = self.fake_A_pool.query(self.fake_A)
        else:
            fake_B = self.fake_B
            fake_A = self.fake_A    
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        return loss_D_A, loss_D_B

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        self.netD_A.requires_grad_(False)
        self.netD_B.requires_grad_(False)        
        g_opt.zero_grad()
        loss_G_A, loss_G_B, loss_G = self.backward_G(batch)
        g_opt.step()
        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)
        d_opt.zero_grad()  
        loss_D_A, loss_D_B = self.backward_D()
        d_opt.step()
        if self.trainer.is_last_batch:
            sch_d.step()
            sch_g.step()
        self.log_dict({"loss_G": loss_G, "loss_G_A": loss_G_A, "loss_G_B":loss_G_B, "loss_D_A":loss_D_A, "loss_D_B":loss_D_B}, prog_bar = True, on_step = True, logger = True)
        return loss_G

    def configure_optimizers(self):
        weight_decay = self.hparams.weight_decay
        g_opt = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr = self.hparams.lr, weight_decay = weight_decay)
        d_opt = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr = self.hparams.lr, weight_decay = weight_decay)
        if self.hparams.lr_scheduler is not None:
            if self.hparams.lr_scheduler == 'step':
                scheduler_g = torch.optim.lr_scheduler.StepLR(g_opt, step_size = self.hparams.lr_decay_steps, gamma = self.hparams.lr_decay_rate)
                scheduler_d = torch.optim.lr_scheduler.StepLR(d_opt, step_size = self.hparams.lr_decay_steps, gamma = self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max = self.hparams.lr_decay_steps, eta_min = self.hparams.lr_decay_min_lr)
                scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max = self.hparams.lr_decay_steps, eta_min = self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
        return [g_opt, d_opt], [{"scheduler": scheduler_g, "interval": "step", "frequency": 1, "name": "lr_vae"}, {"scheduler": scheduler_d, "interval": "step", "frequency": 1, "name": "lr_disc"}]

    def configure_loss(self):
        loss = self.hparams.loss_function.lower()
        if loss == 'mse':
            self.loss_function = torch.nn.functional.mse_loss
        elif loss == 'l1':
            self.loss_function = torch.nn.functional.l1_loss
        elif loss == "bce":
            self.loss_function = torch.nn.functional.binary_cross_entropy
        elif loss == "bcewg":
            self.loss_function = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            raise ValueError("Invalid Loss Type!")