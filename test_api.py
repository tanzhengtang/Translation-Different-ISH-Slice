import SimpleITK as sitk
import torch 
from lightning.pytorch.loggers import WandbLogger

def process_ckp_path(ckpt_path:str, lighting_name_prefix:str = "netG.") -> dict:
    state_dict = torch.load(ckpt_path)
    netG_pth = {}
    for md, mw in state_dict['state_dict'].items():
        if lighting_name_prefix in md:
            netG_pth[md.replace(lighting_name_prefix, "")] = mw
    return netG_pth

if __name__ == "__main__":
    img = sitk.ReadImage("/home/t207/Lab_Data_preproc2/allen_data/code/software/Translation-Different-ISH-Slice/dataset/71249740/ws2dcl6m_39_71111610.png")
    capcit = img.GetSize()[0] * img.GetSize()[1] * 3 / 1024
    print(capcit  / 1024)