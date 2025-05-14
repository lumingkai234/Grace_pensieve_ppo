## Academic Software License: © 2023 UChicago (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-4:
## 
## Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.
## Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.
## IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

import time
import torch
import numpy as np
from .net import load_model, VideoCompressor
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
import torchac
from config_IND import config
from .grace_model import GraceModel, GraceEntropyCoder
import os
import math

class GraceBasicCode:
    def __init__(self, code, shapex, shapey, z, mv_size, res_size):
        self.code = code
        self.shapex = shapex
        self.shapey = shapey
        self.z = z
        self.ipart = None
        self.isize = 0
        self.mv_size = mv_size.detach()
        self.res_size = res_size.detach()
    '''
    def apply_loss(self, loss, no_use=None):
        leng = torch.numel(self.code)

        rnd = torch.rand(leng).to(self.code.device)
        rnd = (rnd > loss).long()
        rnd = rnd[:leng].reshape(self.code.shape)
        self.code = self.code * rnd

        if self.ipart is not None and np.random.random() < loss:
            self.ipart = None
    '''
    def apply_loss(self, loss_ratio, blocksize = 100, use_IND=False, lamda=2048):
        """
        default block size is 100
        """
        #print('Applying loss')
        if not use_IND:
            #print(self.code.shape)
            leng = torch.numel(self.code)
            indices = torch.randperm(leng).to(self.code.device)
            self.code = self.code[indices]
            
            nblocks = (leng - 1) // blocksize + 1

            rnd = torch.rand(nblocks).to(self.code.device)
            rnd = (rnd > loss_ratio).long()
            #print("DEBUG: loss ratio =", loss_ratio, ", first 16 elem:", rnd[:16])
            rnd = rnd.repeat_interleave(blocksize)
            rnd = rnd[:leng].reshape(self.code.shape)
            self.code = self.code * rnd
            self.code = self.code[torch.argsort(indices)]

            #TEST PRINT
            '''
            print(self.mv_size.shape)
            print(self.res_size.shape)
            mv_size_channel = torch.sum(torch.sum(self.mv_size.squeeze(0), dim=2),dim=1)
            res_size_channel = torch.sum(torch.sum(self.res_size.squeeze(0), dim=2),dim=1)
            mv_imp_index, mv_org_index = self.read_importance(lamda, 'mv')
            res_imp_index, res_org_index = self.read_importance(lamda, 'res')
            mv_size_channel_sorted = mv_size_channel[mv_imp_index]
            res_size_channel_sorted = res_size_channel[res_imp_index]
            print(f'MV Channel size {mv_size_channel_sorted}')
            print('#####################################################################')
            print(f'Res Channel size {res_size_channel_sorted}')
            '''
        else:
            #We protect the movtion vector
            mv_size = self.mv_size.squeeze(0)
            res_size = self.res_size.squeeze(0)  #挤出第一个维度
    
            code_mv_quant = self.code[0: self.shapex[0]*self.shapex[1]*self.shapex[2]*self.shapex[3]]
            code_res_quant = self.code[self.shapex[0]*self.shapex[1]*self.shapex[2]*self.shapex[3]:]
            mv_quant = code_mv_quant.reshape(self.shapex).squeeze(0)
            res_quant = code_res_quant.reshape(self.shapey).squeeze(0)
            
            mv_imp_index, mv_org_index = self.read_importance(lamda, 'mv')
            res_imp_index, res_org_index = self.read_importance(lamda, 'res')   #读取重要性排序和原始索引

            code_mv_important = torch.flatten(mv_quant[mv_imp_index][:config.mv_important])
            code_res_important = torch.flatten(res_quant[res_imp_index][:config.res_important])
            mv_size_imp = torch.flatten(mv_size[mv_imp_index][:config.mv_important])
            res_size_imp = torch.flatten(res_size[res_imp_index][:config.res_important])

            mv_size_imp_tot = torch.sum(mv_size_imp)
            res_size_imp_tot = torch.sum(res_size_imp)
            mv_size_unimp_tot = torch.sum(mv_size) - mv_size_imp_tot
            res_size_unimp_tot = torch.sum(res_size) - res_size_imp_tot
            unimportant_size = mv_size_unimp_tot + res_size_unimp_tot
            #unimportant_size_mv = unimportant_size * mv_size_imp_tot / (mv_size_imp_tot + res_size_imp_tot)
            #unimportant_size_res = unimportant_size * res_size_imp_tot / (mv_size_imp_tot + res_size_imp_tot)
            unimportant_size_mv = unimportant_size
            unimportant_size_res = 0
            mv_mask = self.protect_code(code_mv_important, mv_size_imp_tot, unimportant_size_mv, mv_size_imp, 
                                        loss_ratio, config.mv_important, self.shapex, mv_org_index)
            res_mask = self.protect_code(code_res_important, res_size_imp_tot, unimportant_size_res, res_size_imp,
                                        loss_ratio, config.res_important, self.shapey, res_org_index)
            self.code = torch.cat((torch.flatten(mv_mask), torch.flatten(res_mask)), dim=0)
            '''
            reap_times = int(unimportant_size // mv_size_imp_tot)
            #print(f'reap_times = {reap_times}')
            left_size = unimportant_size - reap_times * mv_size_imp_tot
        
            cumulative_sum = torch.cumsum(mv_size_imp, dim=0)
            x = max(torch.searchsorted(cumulative_sum, left_size, right=True).item() - 1, 0)
            #print(reap_times, x)
            #print('left_position = ', x)
            #print(f'Origin size = {torch.sum(mv_size) + torch.sum(res_size)}')
            #print(f'Now size = {(reap_times + 1) * mv_size_imp_tot + torch.sum(mv_size_imp[:x]) + res_size_imp_tot}')
            code_mv_important_1 = self.mask(code_mv_important[:x], loss_ratio ** (reap_times + 2))
            code_mv_important_2 = self.mask(code_mv_important[x:], loss_ratio ** (reap_times + 1))
            code_mv_important = torch.cat((code_mv_important_1, code_mv_important_2), dim=0)
            code_res_important = self.mask(code_res_important, loss_ratio)

            mv_important = code_mv_important.reshape(config.mv_important, self.shapex[2], self.shapex[3])
            res_important = code_res_important.reshape(config.res_important, self.shapey[2], self.shapey[3])
            mv_loss = torch.cat((mv_important, torch.zeros(config.channel_mv - config.mv_important, self.shapex[2], self.shapex[3]).to(mv_important.device)), dim=0)
            res_loss = torch.cat((res_important, torch.zeros(config.channel_res - config.res_important, self.shapey[2], self.shapey[3]).to(res_important.device)), dim=0)
            mv_loss = mv_loss[mv_org_index]
            res_loss = res_loss[res_org_index]
            self.code = torch.cat((torch.flatten(mv_loss), torch.flatten(res_loss)), dim=0)
            '''
    def read_importance(self, lamda, type):
        file_path = f'/home/lmk/Grace_project/IND_Result/lamda_{lamda}/{type}_importance.txt'
        with open(file_path, 'r') as file:
            imp_index = [int(line.strip()) for line in file]
        org_index = sorted(range(len(imp_index)), key=lambda k: imp_index[k])
        return imp_index, org_index
    
    def mask(self, tensor, loss_ratio):
        leng = torch.numel(tensor)
        rnd = torch.rand(leng).to(tensor.device)
        rnd = (rnd > loss_ratio).long()
        tensor_loss = tensor * rnd
        return tensor_loss
    
    def protect_code(self, code_important, important_size, unimportant_size, list_impotant_size, loss_ratio, important_channel, origin_shape, org_index): 
        reap_times = int(unimportant_size // important_size)
        #print(f'reap_times = {reap_times}')
        left_size = unimportant_size - reap_times * important_size
    
        cumulative_sum = torch.cumsum(list_impotant_size, dim=0)
        x = max(torch.searchsorted(cumulative_sum, left_size, right=True).item() - 1, 0)
        
        code_important_1 = self.mask(code_important[:x], loss_ratio ** (reap_times + 2))
        code_important_2 = self.mask(code_important[x:], loss_ratio ** (reap_times + 1))
        code_important_mask = torch.cat((code_important_1, code_important_2), dim=0)

        tensor_important = code_important_mask.reshape(important_channel, origin_shape[2], origin_shape[3])
        tensor_mask = torch.cat((tensor_important, torch.zeros(origin_shape[1] - important_channel, origin_shape[2], origin_shape[3]).to(tensor_important.device)), dim=0)
        tensor_mask = tensor_mask[org_index]

        return tensor_mask
    
class GraceInterface:
    def __init__(self, config, use_half=True, scale_factor=1):
        self.gracemodel = GraceModel(config, use_half, scale_factor)
        self.use_half = use_half

        if use_half:
            self.gracemodel.set_half()
        else:
            self.gracemodel.set_float()

        self.ecmodel = GraceEntropyCoder(self.gracemodel)


    def encode(self, image: torch.Tensor, refer_frame: torch.Tensor) -> GraceBasicCode:
        """
        Main interface of encode
        Input:
            image: torch.tensor with shape 3, h, w, fp32
            refer_frame: torch.tensor with shape 3, h, w, fp32
        Returns:
            code: the GraceBasicCode, can be used for decode and entropy encode
        """
        code, shapex, shapey, z = self.gracemodel.encode(image, refer_frame, return_z = True)

        #estimate the size
        mv_size, res_size = self.ecmodel.entropy_encode(code, shapex, shapey, z, use_estimation=True)[2:4]

        #print(shapex)
        #print(shapey)
        #print(code)
        #print(f"mv shape is {shapex}, residual shape is {shapey}")
        return GraceBasicCode(code, shapex, shapey, z, mv_size, res_size)

    def entropy_encode(self, code: GraceBasicCode):
        """
        Main interface of entropy encode
        Input:
            code: the output of function `encode()`
        Returns:
            a number that is the length of encoded bytestream
        """
        return self.ecmodel.entropy_encode(code.code, code.shapex, code.shapey, code.z, use_estimation=True)[1]

    def decode(self, code: GraceBasicCode, refer_frame: torch.Tensor) -> torch.Tensor:
        """
        Main interface of decode
        Input:
            code: the output of function `encode()`
            refer_frame: the image tensor with shape 3, h, w, fp32
        Returns:
            The decoded image tensor, shape (3, h, w), fp32
        """
        decoded = self.gracemodel.decode(code.code, refer_frame, code.shapex, code.shapey).float().clamp(0, 1)
        return decoded
