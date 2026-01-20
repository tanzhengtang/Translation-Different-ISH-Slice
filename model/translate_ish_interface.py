from cgan_interface import CycleGanInterface
from pixel2pixel_interface import Pixel2PixelInterface
from lightning.pytorch import LightningModule

class IntegratedGANModel(LightningModule):
    def __init__(self, cgan_params, p2p_params, patch_size:int = 256, stride:int = 256):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        self.cgan_model = CycleGanInterface(**cgan_params)
        self.p2p_model = Pixel2PixelInterface(**p2p_params)
        self.patch_size = patch_size
        self.stride = stride

    def filter_patches(self, patches_A, patches_B, threshold=0.1):
        stds = patches_A.view(patches_A.size(0), -1).std(dim=1)
        valid_indices = stds > threshold
        if valid_indices.sum() == 0:
            return None, None
        return patches_A[valid_indices], patches_B[valid_indices]

    def configure_optimizers(self):
        cgan_opt_g, cgan_opt_d = self.cgan_model.optimizers()
        p2p_opt_g, cp2p_opt_d = self.cgan_model.optimizers()
        cgan_sch_g, cgan_sch_d = self.cgan_model.lr_schedulers()
        p2p_sch_g, p2p_sch_d = self.cgan_model.lr_schedulers()
        return [cgan_opt_g, cgan_opt_d, p2p_opt_g, cp2p_opt_d], \
               [{"scheduler": cgan_sch_g, "interval": "step", "frequency": 1, "name": "lr_cgan_g"}, {"scheduler": cgan_sch_d, "interval": "step", "frequency": 1, "name": "lr_cgan_disc"}, {"scheduler": p2p_sch_g, "interval": "step", "frequency": 1, "name": "lr_p2p_g"}, {"scheduler": p2p_sch_d, "interval": "step", "frequency": 1, "name": "lr_p2p_disc"}]

    def tensor_to_patches(self, img_tensor):
        b, c, h, w = img_tensor.shape
        patches = img_tensor.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, self.patch_size, self.patch_size)
        return patches

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch 
        opts = self.optimizers()
        opt_g_cycle, opt_d_cycle = opts[0], opts[1]
        opt_g_p2p, opt_d_p2p = opts[2], opts[3]
        self.cgan_model.real_A = real_A
        self.cgan_model.real_B = real_B
        self.cgan_model.fake_B = self.cgan_model.netG_A(real_A)
        self.cgan_model.rec_A = self.cgan_model.netG_B(self.cgan_model.fake_B)
        self.cgan_model.fake_A = self.cgan_model.netG_B(real_B)
        self.cgan_model.rec_B = self.cgan_model.netG_A(self.cgan_model.fake_A)
        loss_cycle_G = self.compute_cgan_loss_G() 
        opt_g_cycle.zero_grad()
        self.manual_backward(loss_cycle_G)
        opt_g_cycle.step()
        loss_cycle_D = self.compute_cgan_loss_D()
        opt_d_cycle.zero_grad()
        self.manual_backward(loss_cycle_D)
        opt_d_cycle.step()
        fake_B_img = self.cgan_model.fake_B.detach()
        patches_input = self.tensor_to_patches(real_A)     
        patches_target = self.tensor_to_patches(fake_B_img) 
        self.p2p_model.real_X = patches_input
        self.p2p_model.real_Y = patches_target
        self.p2p_model.fake_Y = self.p2p_model.netG(patches_input)
        loss_p2p_G = self.p2p_model.backward_G() 
        opt_g_p2p.zero_grad()
        if not self.p2p_model.automatic_optimization:
             pass 
        else:
             self.manual_backward(loss_p2p_G)
             
        opt_g_p2p.step()
        
    def compute_cgan_loss_G(self):
        m = self.cgan_model
        loss_G_A = m.criterionGAN(m.netD_A(m.fake_B), True)
        loss_G_B = m.criterionGAN(m.netD_B(m.fake_A), True)
        loss_cycle_A = m.criterionCycle(m.rec_A, m.real_A) * 10.0
        loss_cycle_B = m.criterionCycle(m.rec_B, m.real_B) * 10.0
        return loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        
    def compute_cgan_loss_D(self):
        m = self.cgan_model
        loss_D_A = m.backward_D_basic(m.netD_A, m.real_B, m.fake_B)
        loss_D_B = m.backward_D_basic(m.netD_B, m.real_A, m.fake_A)
        return loss_D_A + loss_D_B