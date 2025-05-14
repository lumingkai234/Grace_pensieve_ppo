import os
import torch
from grace.grace_model import GraceModel, GraceEntropyCoder
from grace.net import VideoCompressor
from grace.grace_gpu_interface import GraceInterface
from dataset import DataSet, get_train_loader
from config_IND import config
from tqdm import tqdm
def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

class channel_importance:
    def __init__(self, channel_mv, channel_res, device):
        self.mv_max = torch.full((channel_mv,), -1919810, device=device)
        self.mv_min = torch.full((channel_mv,), 1919810, device=device)
        self.res_max = torch.full((channel_res,), -1919810, device=device)
        self.res_min = torch.full((channel_res,), 1919810, device=device)
        self.channel_mv = channel_mv
        self.channel_res = channel_res

    def update(self, mv_max, mv_min, res_max, res_min):
        self.mv_max = torch.max(mv_max, self.mv_max)
        self.mv_min = torch.min(mv_min, self.mv_min)
        self.res_max = torch.max(res_max, self.res_max)
        self.res_min = torch.min(res_min, self.res_min)
    
    def get_channel_importance(self):
        mv_length = self.mv_max - self.mv_min
        res_length = self.res_max - self.res_min
        mv_sorted_indices = torch.argsort(mv_length, descending=True)
        res_sorted_indices = torch.argsort(res_length, descending=True)
        return mv_sorted_indices, res_sorted_indices

def write_txt(CIND, config):
    mv_max = CIND.mv_max
    mv_min = CIND.mv_min
    res_max = CIND.res_max
    res_min = CIND.res_min
    mv_max = torch.reshape(mv_max, (mv_max.shape[0], 1))
    mv_min = torch.reshape(mv_min, (mv_min.shape[0], 1))
    res_max = torch.reshape(res_max, (res_max.shape[0], 1))
    res_min = torch.reshape(res_min, (res_min.shape[0], 1))
    mv_min_max = torch.cat((mv_min, mv_max), dim=1)
    res_min_max = torch.cat((res_min, res_max), dim=1)
    mv_importance, res_importance = CIND.get_channel_importance()

    file_path_IND = f'./IND_Result/lamda_{config.test_lamda}'
    if not os.path.exists(file_path_IND):
        os.makedirs(file_path_IND)

    mv_min_max_txt = os.path.join(file_path_IND, 'mv_min_max.txt')

    with open(mv_min_max_txt, 'w') as f:
        for i in range(mv_min_max.shape[0]):
            f.write(f'channel_{i:03d}:{mv_min_max[i].tolist()}\n')

    res_min_max_txt = os.path.join(file_path_IND, 'res_min_max.txt')
    with open(res_min_max_txt, 'w') as f:
        for i in range(res_min_max.shape[0]):
            f.write(f'channel_{i:03d}:{res_min_max[i].tolist()}\n')
    
    mv_importance_txt = os.path.join(file_path_IND, 'mv_importance.txt')
    with open(mv_importance_txt, 'w') as f:
        for i in range(mv_importance.shape[0]):
            f.write(f'{mv_importance[i]}\n')

    res_importance_txt = os.path.join(file_path_IND, 'res_importance.txt')
    with open(res_importance_txt, 'w') as f:
        for i in range(res_importance.shape[0]):
            f.write(f'{res_importance[i]}\n')
    
    mv_length = mv_max - mv_min
    mv_length = mv_length[mv_importance]
    mv_length_txt = os.path.join(file_path_IND, 'mv_length.txt')
    with open(mv_length_txt, 'w') as f:
        for i in range(mv_length.shape[0]):
            f.write(f'{mv_length[i][0]}\n')

    res_length = res_max - res_min
    res_length = res_length[res_importance]
    res_length_txt = os.path.join(file_path_IND, 'res_length.txt')
    with open(res_length_txt, 'w') as f:
        for i in range(res_length.shape[0]):
            f.write(f'{res_length[i][0]}\n')

if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    CIND = channel_importance(config.channel_mv, config.channel_res, device)
    train_set = DataSet()
    gpu_num = torch.cuda.device_count()
    train_loader = get_train_loader(train_set, gpu_num, config.gpu_per_batch)
    
    lamda = config.test_lamda
    grace_dvc = VideoCompressor()
    model_path = f'./models/grace/{lamda}_freeze.model'
    load_model(grace_dvc, model_path)
    grace_dvc.to(device)
    grace_dvc.eval()

    if config.use_half:
        grace_dvc.half()

    scale_factor_dic = {
        64 : 0.25,
        128 : 0.5,
        256 : 0.5,
        512 : 0.5,
        1024 : 0.5,
        2048 : 1.0,
        4096 : 1.0,
        8192 : 1.0,
        12288 : 1.0,
        16384 : 1.0,
    }
    grace_dvc.set_scale_factor(scale_factor_dic[lamda])
    print(f'Finding In-Distribution of each channel for model {lamda}:')
    with torch.no_grad():
        with tqdm(total=len(train_loader), unit='batch') as pbar:
            for batch_idx, data in enumerate(train_loader):
                input_image, ref_image = data[0].to(device), data[1].to(device)
                quant_noise_feature, quant_noise_z, quant_noise_mv = data[2].to(device), data[3].to(device), data[4].to(device)
                if config.get_IND:
                    quant_mv, compressed_feature_renorm = grace_dvc(input_image, ref_image, 
                                                                    quant_noise_feature, quant_noise_z, 
                                                                    quant_noise_mv, get_IND=config.get_IND)
                else: #Test the pretrained DVC model is available
                    clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = grace_dvc(input_image, ref_image, 
                                                                                                                    quant_noise_feature, quant_noise_z, 
                                                                                                                    quant_noise_mv, get_IND=config.get_IND)
                    mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
                                                        torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv), torch.mean(bpp)
                    print(f'batch_idx = {batch_idx} || bpp: {bpp}, mse_loss: {mse_loss}, warploss: {warploss}, interloss: {interloss}')
                    continue
                mv_max = torch.amax(quant_mv, dim=(0, 2, 3))
                mv_min = torch.amin(quant_mv, dim=(0, 2, 3))
                res_max = torch.amax(compressed_feature_renorm, dim=(0, 2, 3))
                res_min = torch.amin(compressed_feature_renorm, dim=(0, 2, 3))
                CIND.update(mv_max, mv_min, res_max, res_min)
                pbar.update(1)
                pbar.set_postfix({"batch_idx": batch_idx})
                #print(CIND.mv_max)
            write_txt(CIND, config)
        
        
                