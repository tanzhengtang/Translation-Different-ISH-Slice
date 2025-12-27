from model import networks

class Pixel2PixelInterface(networks.GanCommonModel):
    def __init__(self,):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.load_networks()
        self.configure_loss()

    def load_networks(self):
        self.netG = networks.NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.netD = networks.NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        if hasattr(self.hparams, 'netG_ckpt_path'):
            self.netG.load_from_ckpt(self.hparams.netG_ckpt_path)
        if hasattr(self.hparams, 'netD_ckpt_path'):
            self.netD.load_from_ckpt(self.hparams.netD_ckpt_path)

    def backward_G(self):
        loss_G = self.criterionGAN(self.netD(self.fake_Y), True)
        loss_G.backward()
        return loss_G

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D(self):
        loss_D = self.backward_D_basic(self.netD, self.real_Y, self.fake_Y)
        return loss_D
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        self.real_X = x
        self.real_Y = y
        self.fake_Y = self.netG(self.real_X)
        g_opt, d_opt = self.optimizers()
        g_opt.zero_grad()
        self.netD.requires_grad_(False)
        loss_G = self.backward_G()
        g_opt.step()
        self.netD.requires_grad_(True)
        d_opt.zero_grad()
        loss_D = self.backward_D()
        d_opt.step()
        self.log_dict({"loss_G": loss_G, "loss_D": loss_D}, prog_bar = True, on_step = True, logger = True)
        return loss_G
