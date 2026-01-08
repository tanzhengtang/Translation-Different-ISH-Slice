from model import networks
import torch

class Pixel2PixelInterface(networks.GanCommonModel):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.load_networks()
        self.configure_loss()

    def load_networks(self):
        self.netG = networks.NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.hparams.netD_params['input_nc'] = self.hparams.netG_params['input_nc'] + self.hparams.netG_params['output_nc']
        self.netD = networks.NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        if hasattr(self.hparams, 'netG_ckpt_path'):
            self.netG.load_from_ckpt(self.hparams.netG_ckpt_path)
        if hasattr(self.hparams, 'netD_ckpt_path'):
            self.netD.load_from_ckpt(self.hparams.netD_ckpt_path)

    def backward_G(self):
        fake_XY = torch.cat((self.real_X, self.fake_Y), 1)
        pred_fake = self.netD(fake_XY)        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_L1 = torch.nn.functional.l1_loss(self.fake_Y, self.real_Y) * self.hparams.get('lambda_L1', 100.0)
        loss_G = loss_G_GAN + loss_G_L1
        self.manual_backward(loss_G)
        return loss_G

    def backward_D(self):
        fake_XY = torch.cat((self.real_X, self.fake_Y), 1)
        pred_fake = self.netD(fake_XY.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        real_XY = torch.cat((self.real_X, self.real_Y), 1)
        pred_real = self.netD(real_XY)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.manual_backward(loss_D)
        return loss_D
    
    def training_step(self, batch, batch_idx):
        self.real_X, self.real_Y = batch
        self.fake_Y = self.netG(self.real_X)
        g_opt, d_opt = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        self.netD.requires_grad_(True)
        d_opt.zero_grad()
        loss_D = self.backward_D()
        d_opt.step()
        sch_d.step()
        self.netD.requires_grad_(False)
        g_opt.zero_grad()
        loss_G = self.backward_G()
        g_opt.step()
        sch_g.step() 
        self.log_dict({"loss_G": loss_G, "loss_D": loss_D, 'lr_g': g_opt.param_groups[0]['lr'], 'lr_d':d_opt.param_groups[0]['lr']}, on_epoch = False, prog_bar = True, on_step = True, logger = True, sync_dist = True)
        return loss_G
