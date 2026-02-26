from model import cgan_interface
import torch

class IntegratedGANModel(cgan_interface.CycleGanInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return self.cgan_model(x)

    def configure_optimizers(self):
        cgan_conf = self.cgan_model.configure_optimizers() 
        p2p_conf = self.p2p_model.configure_optimizers()
        cgan_sch_g, cgan_sch_d = self.cgan_model.lr_schedulers()
        p2p_sch_g, p2p_sch_d = self.cgan_model.lr_schedulers()
        return cgan_conf + p2p_conf, \
               [{"scheduler": cgan_sch_g, "interval": "step", "frequency": 1, "name": "lr_cgan_g"}, {"scheduler": cgan_sch_d, "interval": "step", "frequency": 1, "name": "lr_cgan_disc"}, {"scheduler": p2p_sch_g, "interval": "step", "frequency": 1, "name": "lr_p2p_g"}, {"scheduler": p2p_sch_d, "interval": "step", "frequency": 1, "name": "lr_p2p_disc"}]

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

    def training_step(self, batch, batch_idx):
        loss_list =[]
        cgan_g_opt, cgan_d_opt, p2p_g_opt, p2p_d_opt = self.optimizers()
        cgan_sch_g, cgan_sch_d, p2p_sch_g, p2p_sch_d = self.lr_schedulers()
        self.cgan_model.netD_A.requires_grad_(False)
        self.cgan_model.netD_B.requires_grad_(False)        
        cgan_g_opt.zero_grad()
        loss_cgan_G_A, _, _ = self.cgan_model.backward_G(batch)
        cgan_g_opt.step()
        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)
        cgan_d_opt.zero_grad()  
        loss_cgan_D_A, _ = self.cgan_model.backward_D()
        cgan_d_opt.step()
        if self.trainer.is_last_batch:
            cgan_sch_g.step()
            cgan_sch_d.step()
        self.log_dict({"loss_cgan_G_A": loss_cgan_G_A, "loss_cgan_D_A": loss_cgan_D_A}, prog_bar = True, on_step = True, logger = True)
        loss_list.append(loss_cgan_G_A, loss_cgan_D_A)
        
        if self.current_epoch > self.p2p_start_train_epoch:
            fake_B = self.cgan_model.fake_B.detach()
            patches_real_A = self.tensor_to_patches(batch[0])
            patches_fake_B = self.tensor_to_patches(fake_B)
            valid_real_A, valid_fake_B = self.filter_patches(patches_real_A, patches_fake_B) 
            if valid_real_A is not None:
                self.p2p_model.netD.requires_grad_(False)
                p2p_g_opt.zero_grad()
                loss_p2p_G = self.p2p_model.backward_G([valid_fake_B, valid_real_A]) # the direction must be invert, since we want to get the B_g1 -> A_g1 for anthoer B_g2 -> A_g1
                p2p_g_opt.step()
                self.p2p_model.netD.requires_grad_(True)
                p2p_d_opt.zero_grad()
                loss_p2p_D = self.p2p_model.backward_D()
                p2p_d_opt.step()
            if self.trainer.is_last_batch:
                p2p_sch_g.step()
                p2p_sch_d.step()
            self.log_dict({"loss_p2p_G": loss_p2p_G, "loss_p2p_D": loss_p2p_D}, prog_bar = True, on_step = True, logger = True)
        return loss_cgan_G_A