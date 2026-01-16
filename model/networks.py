import torch
import typing

def process_ckp_path(ckpt_path:str, lighting_name_prefix:str = "netG.") -> dict:
    state_dict = torch.load(ckpt_path)
    netG_pth = {}
    for md, mw in state_dict['state_dict'].items():
        if lighting_name_prefix in md:
            netG_pth[md.replace(lighting_name_prefix, "")] = mw
    return netG_pth

def get_3Dnorm_layer(norm_layer:str, dim:int) -> torch.nn.Module:
    if norm_layer == "instance":
        norm_layer = torch.nn.InstanceNorm3d(dim)
    elif norm_layer == "batch":
        norm_layer = torch.nn.BatchNorm3d(dim)
    elif norm_layer == "None":
        norm_layer == torch.nn.Identity()
    else:
        raise(f"No such {norm_layer} norm layer")
    return norm_layer

def get_2Dnorm_layer(norm_layer:str, dim:int) -> torch.nn.Module:
    if norm_layer == "instance":
        norm_layer = torch.nn.InstanceNorm2d(dim) 
    elif norm_layer == "batch":
        norm_layer = torch.nn.BatchNorm2d(dim)
    else:
        raise(f"No such {norm_layer} norm layer")
    return norm_layer

class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    random_id = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

class NLayerDiscriminator3D(torch.nn.Module):
    def __init__(self, input_nc:int, ndf:int = 64, n_layers:int = 3, kernel_size:int = 4, norm_layer:str = "batch", use_sigmoid:bool = False, use_bias:bool = False, padding:int = 1, stride:int = 2):
        super(NLayerDiscriminator3D, self).__init__()
        sequence = [torch.nn.Conv3d(input_nc, ndf, kernel_size = kernel_size, stride = stride, padding = padding), torch.nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [torch.nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kernel_size, stride = stride, padding = padding, bias = use_bias), get_3Dnorm_layer(norm_layer, ndf * nf_mult), torch.nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [torch.nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kernel_size, stride = 1, padding = padding, bias = use_bias), get_3Dnorm_layer(norm_layer, ndf * nf_mult), torch.nn.LeakyReLU(0.2, True)]
        sequence += [torch.nn.Conv3d(ndf * nf_mult, 1, kernel_size = kernel_size, stride = 1, padding = padding)]
        if use_sigmoid:
            sequence.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator2D(torch.nn.Module):
    def __init__(self, input_nc:int, ndf:int = 64, n_layers:int = 3, kernel_size:int = 4, norm_layer:str = "batch", use_sigmoid:bool = False, use_bias:bool = False, padding:int = 1, stride:int = 2):
        super(NLayerDiscriminator2D, self).__init__()
        sequence = [torch.nn.Conv2d(input_nc, ndf, kernel_size = kernel_size, stride = stride, padding = padding), torch.nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kernel_size, stride = stride, padding = padding, bias = use_bias), get_2Dnorm_layer(norm_layer, ndf * nf_mult), torch.nn.LeakyReLU(True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kernel_size, stride = 1, padding = padding, bias = use_bias), get_2Dnorm_layer(norm_layer, ndf * nf_mult), torch.nn.LeakyReLU(True)]
        sequence += [torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size = kernel_size, stride = 1, padding = padding)]
        if use_sigmoid:
            sequence.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class SpectralNormalizationDiscriminator2D(torch.nn.Module):
    def __init__(self, input_nc:int, ndf:int = 64, n_layers:int = 3, kernel_size:int = 4, use_sigmoid:bool = False, use_bias:bool = False, padding:int = 1, stride:int = 2):
        super(SpectralNormalizationDiscriminator2D, self).__init__()
        sequence = [torch.nn.utils.spectral_norm(torch.nn.Conv2d(input_nc, ndf, kernel_size = kernel_size, stride = stride, padding = padding)), torch.nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [torch.nn.utils.spectral_norm(torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kernel_size, stride=stride, padding = padding, bias = use_bias)), torch.nn.LeakyReLU(True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [torch.nn.utils.spectral_norm(torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kernel_size, stride = 1, padding = padding, bias = use_bias)), torch.nn.LeakyReLU(True)]
        sequence += [torch.nn.utils.spectral_norm(torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size = kernel_size, stride = 1, padding = padding))]
        if use_sigmoid:
            sequence.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*sequence)
            
    def forward(self, input):
        return self.model(input)

class Dense3DLayer(torch.nn.Module):
    def __init__(self, input_features, output_features, norm_layer = "batch", kernel_size = 1, padding = 0, drop_rate = 0.):
        super(Dense3DLayer, self).__init__()
        self.model = torch.nn.Sequential(*[get_3Dnorm_layer(norm_layer, input_features), torch.nn.ReLU(True), torch.nn.Conv3d(input_features, output_features, kernel_size = kernel_size, stride = 1, padding = padding, bias = False)])
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        x = self.model(x)
        if self.drop_rate > 0:
            x = torch.nn.functional.dropout(x, p = self.drop_rate, training = self.training)
        return x

class Dense3DBlock(torch.nn.Module):
    def __init__(self, input_nc, bn_size, growth_rate, norm_layer = "batch", kernel_size = 1, padding = 0, drop_rate=0.):
        super(Dense3DBlock, self).__init__()
        model_list = []
        model_list.append(Dense3DLayer(input_features = input_nc, output_features = bn_size, norm_layer = norm_layer, kernel_size = kernel_size, padding = padding, drop_rate = drop_rate))
        model_list.append(Dense3DLayer(input_features = bn_size, output_features = growth_rate, norm_layer = norm_layer, kernel_size = kernel_size, padding = padding, drop_rate = drop_rate))
        self.model = torch.nn.Sequential(*model_list)

    def forward(self, x):
        x = torch.cat((x, self.model(x)), dim=1)
        return x

class DenseTransition3D(torch.nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer = "batch"):
        super(DenseTransition3D, self).__init__()
        self.model = torch.nn.Sequential(*[get_3Dnorm_layer(norm_layer, input_nc), torch.nn.Conv3d(input_nc, output_nc, kernel_size = 1, stride = 1, bias = False), torch.nn.AvgPool3d(kernel_size = 2, stride = 2)])
    
    def forward(self, x):
        return self.model(x)

class DesnetGenerator3D(torch.nn.Module):
    def __init__(self, input_nc:int, output_nc:int, ngf:int = 64, norm_layer:typing.Literal["instance", "batch"] = "batch", n_blocks:int = 3, padding_mode:str = "zeros", use_bias:bool = True, downsample_times:int = 2, drop_rate:float = 0):
        super(DesnetGenerator3D, self).__init__()
        model_list = [self.downsample_block(input_nc, ngf, padding_mode, 7, 3, norm_layer, 1, use_bias)]
        for i in range(downsample_times):
            mult = 2**i
            model_list.append(self.downsample_block(ngf * mult, ngf * mult * 2, "zeros", 3, 1, norm_layer, 2, use_bias))

        num_featers = ngf * mult * 2
        for i in range(n_blocks):
            new_num_featers = num_featers + ngf * i
            model_list.append(Dense3DBlock(new_num_featers, new_num_featers // 2, ngf, norm_layer, 3, 1, drop_rate))
        new_num_featers = new_num_featers + ngf
        model_list.append(Dense3DLayer(new_num_featers, new_num_featers // 2, norm_layer, 3, 1, drop_rate))
        model_list.append(Dense3DLayer(new_num_featers // 2, ngf * mult * 2, norm_layer, 3, 1, drop_rate))

        for i in range(downsample_times):
            mult = 2**(downsample_times - i)
            model_list.append(self.upsample_block(ngf * mult, int((ngf * mult / 2)), "zeros", 3, norm_layer, 2, use_bias))
        model_list.append(torch.nn.Conv3d(ngf, output_nc, 7, 1, 3, 1, 1, use_bias, padding_mode))
        model_list.append(torch.nn.Tanh())
        self.model = torch.nn.Sequential(*model_list)
        
    def downsample_block(self, input_nc:int, output_nc:int, padding_mode:str, kernel_size:int, pading:int = 1, norm_layer:str = "instance", stride:int = 1, use_bias:bool = True):
        model_list = [torch.nn.Conv3d(input_nc, output_nc, kernel_size, stride, pading, 1, 1, use_bias, padding_mode), get_3Dnorm_layer(norm_layer, output_nc),torch.nn.ReLU(True)]
        return torch.nn.Sequential(*model_list)
    
    def upsample_block(self, input_nc:int, output_nc:int, padding_mode:str, kernel_size:int, norm_layer:str = "instance", stride:int = 1, use_bias:bool = True):
        model_list = [torch.nn.ConvTranspose3d(input_nc, output_nc, kernel_size, stride, 1, 1, 1, use_bias, 1, padding_mode), get_3Dnorm_layer(norm_layer, output_nc), torch.nn.ReLU(True)]
        return torch.nn.Sequential(*model_list)

    def forward(self, x):
        if len(x.shape) == 4:
            return torch.squeeze(self.model(torch.unsqueeze(x, 0)), 0)
        return self.model(x)

    def load_from_ckpt(self, ckpt_path:str, lighting_name_prefix:str = "netG."):
        self.load_state_dict(process_ckp_path(ckpt_path, lighting_name_prefix))

class ResnetBlock3D(torch.nn.Module):
    def __init__(self, dim:int, padding_type:str = "zeros", kernel_size:int = 3, norm_layer:str = "instance", drop_rate:int = 0.5, use_bias:bool = True):
        super(ResnetBlock3D, self).__init__()
        model_list = [torch.nn.Conv3d(dim, dim, kernel_size, 1, 1, 1, 1, use_bias, padding_type)]
        model_list.append(get_3Dnorm_layer(norm_layer, dim))
        if drop_rate:
            model_list.append(torch.nn.Dropout3d(drop_rate))
        model_list.append(torch.nn.ReLU(True))
        model_list.append(torch.nn.Conv3d(dim, dim, kernel_size, 1, 1, 1, 1, use_bias, padding_type))
        self.model = torch.nn.Sequential(*model_list)

    def forward(self, x):
        return x + self.model(x)

class ResnetGenerator3D(torch.nn.Module):
    def __init__(self, input_nc:int, output_nc:int, ngf:int = 64, norm_layer:typing.Literal["instance", "batch"] = "instance", drop_rate:int = 0, n_blocks:int = 6, padding_mode:str = "zeros", use_bias:bool = True, downsample_times:int = 2):
        super(ResnetGenerator3D, self).__init__()
        model_list = [self.downsample_block(input_nc, ngf, padding_mode, 7, 3, norm_layer, 1, False)]
        for i in range(downsample_times):
            mult = 2**i
            model_list.append(self.downsample_block(ngf * mult, ngf * mult * 2, "zeros", 3, 1, norm_layer, 2, False))
        for i in range(n_blocks):
            model_list.append(ResnetBlock3D(ngf * (2**downsample_times), padding_mode, 3, norm_layer, drop_rate, use_bias))
        for i in range(downsample_times):
            mult = 2**(downsample_times - i)
            model_list.append(self.upsample_block(ngf * mult, int((ngf * mult / 2)), "zeros", 3, norm_layer, 2, False))
        model_list.append(torch.nn.Conv3d(ngf, output_nc, 7, 1, 3, 1, 1, use_bias, padding_mode))
        model_list.append(torch.nn.Tanh())
        self.model = torch.nn.Sequential(*model_list)
        
    def downsample_block(self, input_nc:int, output_nc:int, padding_mode:str, kernel_size:int, pading:int = 1, norm_layer:str = "instance", stride:int = 1, use_bias:bool = True):
        model_list = [torch.nn.Conv3d(input_nc, output_nc, kernel_size, stride, pading, 1, 1, use_bias, padding_mode), get_3Dnorm_layer(norm_layer, output_nc),torch.nn.ReLU(True)]
        return torch.nn.Sequential(*model_list)
    
    def upsample_block(self, input_nc:int, output_nc:int, padding_mode:str, kernel_size:int, norm_layer:str = "instance", stride:int = 1, use_bias:bool = True):
        model_list = [torch.nn.ConvTranspose3d(input_nc, output_nc, kernel_size, stride, 1, 1, 1, use_bias, 1, padding_mode), get_3Dnorm_layer(norm_layer, output_nc), torch.nn.ReLU(True)]
        return torch.nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

    def load_from_ckpt(self, ckpt_path:str, lighting_name_prefix:str = "netG."):
        self.load_state_dict(process_ckp_path(ckpt_path, lighting_name_prefix))

class ResnetBlock2D(torch.nn.Module):
    def __init__(self, dim:int, padding_type:str = "zeros", kernel_size:int = 3, norm_layer:str = "instance", drop_rate:int = 0.5, use_bias:bool = True):
        super(ResnetBlock2D, self).__init__()
        model_list = [torch.nn.Conv2d(dim, dim, kernel_size, 1, 1, 1, 1, use_bias, padding_type)]
        model_list.append(get_2Dnorm_layer(norm_layer, dim))
        if drop_rate:
            model_list.append(torch.nn.Dropout2d(drop_rate))
        model_list.append(torch.nn.ReLU(True))
        model_list.append(torch.nn.Conv2d(dim, dim, kernel_size, 1, 1, 1, 1, use_bias, padding_type))
        self.model = torch.nn.Sequential(*model_list)

    def forward(self, x):
        return x + self.model(x)

class ResnetGenerator2D(torch.nn.Module):
    def __init__(self, input_nc:int, output_nc:int, ngf:int = 64, norm_layer:typing.Literal["instance", "batch"] = "instance", drop_rate:int = 0, n_blocks:int = 6, padding_mode:str = "zeros", use_bias:bool = True, downsample_times:int = 2):
        super(ResnetGenerator2D, self).__init__()
        model_list = [self.downsample_block(input_nc, ngf, padding_mode, 7, 3, norm_layer, 1, False)]
        for i in range(downsample_times):
            mult = 2**i
            model_list.append(self.downsample_block(ngf * mult, ngf * mult * 2, "zeros", 3, 1, norm_layer, 2, False))
        for i in range(n_blocks):
            model_list.append(ResnetBlock2D(ngf * (2**downsample_times), padding_mode, 3, norm_layer, drop_rate, use_bias))
        for i in range(downsample_times):
            mult = 2**(downsample_times - i)
            model_list.append(self.upsample_block(ngf * mult, int((ngf * mult / 2)), "zeros", 3, norm_layer, 2, False))
        model_list.append(torch.nn.Conv2d(ngf, output_nc, 7, 1, 3, 1, 1, use_bias, padding_mode))
        model_list.append(torch.nn.Tanh())
        self.model = torch.nn.Sequential(*model_list)
        
    def downsample_block(self, input_nc:int, output_nc:int, padding_mode:str, kernel_size:int, pading:int = 1, norm_layer:str = "instance", stride:int = 1, use_bias:bool = True):
        model_list = [torch.nn.Conv2d(input_nc, output_nc, kernel_size, stride, pading, 1, 1, use_bias, padding_mode), get_2Dnorm_layer(norm_layer, output_nc),torch.nn.ReLU(True)]
        return torch.nn.Sequential(*model_list)
    
    def upsample_block(self, input_nc:int, output_nc:int, padding_mode:str, kernel_size:int, norm_layer:str = "instance", stride:int = 1, use_bias:bool = True):
        model_list = [torch.nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, 1, 1, 1, use_bias, 1, padding_mode), get_2Dnorm_layer(norm_layer, output_nc), torch.nn.ReLU(True)]
        return torch.nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

    def load_from_ckpt(self, ckpt_path:str, lighting_name_prefix:str = "netG."):
        self.load_state_dict(process_ckp_path(ckpt_path, lighting_name_prefix))

NETWORKS_CLASS_DICT = dict(ResnetGenerator2D = ResnetGenerator2D,
                ResnetGenerator3D = ResnetGenerator3D,
                NLayerDiscriminator2D = NLayerDiscriminator2D,
                NLayerDiscriminator3D = NLayerDiscriminator3D,
                DesnetGenerator3D = DesnetGenerator3D,
                SpectralNormalizationDiscriminator2D = SpectralNormalizationDiscriminator2D)

from lightning.pytorch import LightningModule
import torchmetrics

METRICS_CLASS_DICT = dict(MSE = torchmetrics.functional.mean_squared_error,
                          MAE = torchmetrics.functional.mean_absolute_error,
                          PSNR = torchmetrics.functional.peak_signal_noise_ratio,
                          SSIM = torchmetrics.functional.structural_similarity_index_measure,
                          MS_SSIM = torchmetrics.functional.multiscale_structural_similarity_index_measure)

class GanCommonModel(LightningModule):
    def __init__(self, netG_name:str, netD_name:str, netG_params:dict, netD_params:dict, loss_function:str, weight_decay:float, lr:float, lr_scheduler:str, lr_decay_steps:float, lr_decay_min_lr:float, lr_decay_rate:float, val_metric_names:list = []):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.load_networks()
        self.configure_loss()
    
    def forward(self, x):
        return self.netG(x)
    
    def criterionGAN(self, input_tensor:torch.Tensor, target_is_real:bool, use_smoothing:bool = False):
        if use_smoothing:
            if target_is_real:
                target_tensor = 0.7 + 0.5 * torch.rand_like(input_tensor, device=self.device)
            else:
                target_tensor = 0.1 * torch.rand_like(input_tensor, device=self.device)
        else:
            if target_is_real:
                target_tensor = torch.ones_like(input_tensor, device=self.device)
            else:
                target_tensor = torch.zeros_like(input_tensor, device=self.device)
        return self.loss_function(input_tensor, target_tensor)
    
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
    
    def configure_optimizers(self):
        weight_decay = self.hparams.get('weight_decay', 0)
        g_opt = torch.optim.Adam(self.netG.parameters(), lr = self.hparams.lr, weight_decay = weight_decay)
        d_opt = torch.optim.Adam(self.netD.parameters(), lr = self.hparams.lr * 0.5, weight_decay = weight_decay)
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
    
    def load_networks(self):
        self.netG = NETWORKS_CLASS_DICT[self.hparams.netG_name](**self.hparams.netG_params)
        self.netD = NETWORKS_CLASS_DICT[self.hparams.netD_name](**self.hparams.netD_params)
        if hasattr(self.hparams.netG_params, 'netG_ckpt_path'):
            self.netG.load_from_ckpt(self.hparams.netG_params['netG_ckpt_path'])
        if hasattr(self.hparams.netD_params, 'netD_ckpt_path'):
            self.netD.load_from_ckpt(self.hparams.netD_params['netD_ckpt_path'])
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        g_opt, d_opt = self.optimizers()
        g_y = self.netG(x)
        g_opt.zero_grad()
        self.netD.requires_grad_(False)
        errG = torch.nn.functional.l1_loss(g_y, y)
        self.manual_backward(errG)
        g_opt.step()
        self.netD.requires_grad_(True)
        d_opt.zero_grad()
        d_y = self.netD(y)
        d_g_y = self.netD(g_y.detach())
        errD_real = self.criterionGAN(d_y, is_real = True)
        errD_fake = self.criterionGAN(d_g_y, is_real = False)
        errD = (errD_real + errD_fake) * 0.5
        self.manual_backward(errD)
        d_opt.step()
        self.log_dict({"g_loss": errG, "d_loss": errD, "errD_real":errD_real, "errD_fake":errD_fake}, prog_bar = True, on_step = True, logger = True)
        return errG
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_g_y = self.netG(x)
        val_loss_dict = {}
        if hasattr(self.hparams, 'val_metric_names'):
            for metric_name in self.hparams.val_metric_names:
                val_loss_dict[metric_name] = METRICS_CLASS_DICT[metric_name](val_g_y.detach(), y)
        if val_loss_dict:
            self.log_dict(val_loss_dict, prog_bar = True, on_step = True, logger = True, sync_dist = True)
        return val_g_y.detach()
    
    def test_step(self, batch, batch_idx):
        return