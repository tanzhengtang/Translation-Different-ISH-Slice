import torch
import itertools
from model import networks
import lightning

class CycleGanInterface(networks.GanCommonModel):
    def __init__(self, netG_name:str, netD_name:str, netG_params:dict, netD_params:dict, loss_function:str, weight_decay:float, lr:float, lr_scheduler:str, lr_decay_steps:float, lr_decay_min_lr:float, lr_decay_rate:float, val_metric_names:list, netG_ckpt_path:str, netD_ckpt_path:str):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.load_networks()
        self.configure_loss()

    def load_networks(self):
        self.netG_A = networks.NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.netG_B = networks.NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.netD_A = networks.NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        self.netD_B = networks.NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        if hasattr(self.hparams.netG_params, 'netG_A_ckpt_path'):
            self.netG_A.load_from_ckpt(self.hparams.netG_A_ckpt_path)
        if hasattr(self.hparams.netG_params, 'netG_B_ckpt_path'):
            self.netG_B.load_from_ckpt(self.hparams.netG_B_ckpt_path)
        if hasattr(self.hparams.netD_params, 'netD_A_ckpt_path'):
            self.netD_A.load_from_ckpt(self.hparams.netD_A_ckpt_path)
        if hasattr(self.hparams.netD_params, 'netD_B_ckpt_path'):
            self.netD_B.load_from_ckpt(self.hparams.netD_B_ckpt_path)
        pool_image_size = self.hparams.get('pool_image_size', 50)
        self.fake_A_pool = networks.ImagePool(pool_image_size)
        self.fake_B_pool = networks.ImagePool(pool_image_size)
        self.direction = self.hyparams.get('direction', 'AtoB')
        if self.direction == 'AtoB':
            self.netG = self.netG_A
        else:
            self.netG = self.netG_B

    def backward_G(self):
        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0
        criterionIdt = torch.nn.L1Loss()
        criterionCycle = torch.nn.L1Loss()
        # Identity loss
        if lambda_idt > 0:
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        loss_cycle_A = criterionCycle(self.rec_A, self.real_A) * lambda_A
        loss_cycle_B = criterionCycle(self.rec_B, self.real_B) * lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        return loss_G_A, loss_G_B, loss_G

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
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
        self.real_A, self.real_B = batch
        g_opt, d_opt = self.optimizers()
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        self.netD_A.requires_grad_(False)
        self.netD_B.requires_grad_(False)        
        g_opt.zero_grad()
        loss_G_A, loss_G_B, loss_G = self.backward_G()
        g_opt.step()

        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)
        d_opt.zero_grad()  
        loss_D_A, loss_D_B = self.backward_D()
        d_opt.step()

        self.log_dict({"loss_G": loss_G, "loss_G_A": loss_G_A, "loss_G_B":loss_G_B, "loss_D_A":loss_D_A, "loss_D_B":loss_D_B}, prog_bar = True, on_step = True, logger = True)
        return loss_G

    def configure_optimizers(self):
        weight_decay = self.hparams.weight_decay
        g_opt = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr = self.hparams.lr, weight_decay = weight_decay)
        d_opt = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr = self.hparams.lr, weight_decay = weight_decay)
        if self.hparams.lr_scheduler is not None:
            if self.hparams.lr_scheduler == 'step':
                torch.optim.lr_scheduler.StepLR(g_opt, step_size = self.hparams.lr_decay_steps, gamma = self.hparams.lr_decay_rate)
                torch.optim.lr_scheduler.StepLR(d_opt, step_size = self.hparams.lr_decay_steps, gamma = self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max = self.hparams.lr_decay_steps, eta_min = self.hparams.lr_decay_min_lr)
                torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max = self.hparams.lr_decay_steps, eta_min = self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
        return g_opt, d_opt

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