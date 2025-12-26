import inspect
import torch
import importlib
import itertools
from lightning.pytorch import LightningModule
import torchmetrics

class CycleGanInterface(LightningModule):
    def __init__(self, netG_name:str, netD_name:str, netG_params:dict, netD_params:dict, loss_function:str, weight_decay:float, lr:float, lr_scheduler:str, lr_decay_steps:float, lr_decay_min_lr:float, lr_decay_rate:float, direction:str):
        super().__init__()
        self.automatic_optimization = False
        self.netG_A = torch.nn.LazyBatchNorm1d()
        self.netD_A = torch.nn.LazyBatchNorm1d()
        self.netG_B = torch.nn.LazyBatchNorm1d()
        self.netD_B = torch.nn.LazyBatchNorm1d()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, x):
        return self.netG_A(x)

    def criterionGAN(self, input_tensor, is_real = True):
        target_tensor = torch.zeros_like(input_tensor, device = self.device) if not is_real else torch.ones_like(input_tensor, device = self.device)
        return self.loss_function(input_tensor, target_tensor)

    def Cor_CoeLoss(self, y_pred, y_target):
        x = y_pred
        y = y_target
        x_var = x - torch.mean(x)
        y_var = y - torch.mean(y)
        r_num = torch.sum(x_var * y_var)
        r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
        r = r_num / r_den
        # return 1 - r  # best are 0
        return 1 - r**2 # abslute constrain

    def backward_G(self):
        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0

        criterionIdt = torch.nn.L1Loss()
        criterionCycle = torch.nn.L1Loss()
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss
        loss_cycle_A = criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        loss_cycle_B = criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss
        # lambda_co_A = 2
        # lambda_co_B = 2
        # loss_cor_coe_GA = self.Cor_CoeLoss(self.fake_B, self.real_A) * lambda_co_A  # fake ct & real mr; Evaluate the Generator of ct(G_A)
        # loss_cor_coe_GB = self.Cor_CoeLoss(self.fake_A, self.real_B) * lambda_co_B  # fake mr & real ct; Evaluate the Generator of mr(G_B)
        # loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_cor_coe_GA + loss_cor_coe_GB
        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cor_coe_GA + self.loss_cor_coe_GB
        loss_G.backward()
        return loss_G_A, loss_G_B, loss_G

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_ssim_A = torchmetrics.functional.multiscale_structural_similarity_index_measure(self.netG_A(x).detach(), y, data_range = 1.0)
        val_ssim_B = torchmetrics.functional.multiscale_structural_similarity_index_measure(self.netG_B(y).detach(), x, data_range = 1.0)
        self.log_dict({"val_G_A": val_ssim_A, "val_G_B": val_ssim_B}, prog_bar = True, on_step = True, logger = True)
        val_ssim = val_ssim_A if self.direction == "AtoB" else val_ssim_B
        return val_ssim

    def test_step(self, batch, batch_idx):
        return

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

    def load_model(self):
        netG_name = self.hparams.netG_name
        netD_name = self.hparams.netD_name
        Generator_Module_Name = "Generator"
        Discriminator_Module_Name = "Discriminator"
        
        try:
            netG = getattr(importlib.import_module(f".{Generator_Module_Name}", package= ".model"), netG_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {Generator_Module_Name}.{netG_name}!')
        try:
            netD = getattr(importlib.import_module(f".{Discriminator_Module_Name}", package= ".model"), netD_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {Discriminator_Module_Name}.{netD_name}!')
        self.netG_A = self.instancialize(netG, **self.hparams.netG_params)
        self.netG_B = self.instancialize(netG, **self.hparams.netG_params)
        self.netD_A = self.instancialize(netD, **self.hparams.netD_params)
        self.netD_B = self.instancialize(netD, **self.hparams.netD_params)
 
    def instancialize(self, Model, **other_args):
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)