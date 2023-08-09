#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(r'D:\project\paper_flatten\code\integrated_code\bin\lama')

from torch.utils.data import Dataset
from saicinpainting.evaluation.data import scale_image, pad_img_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

class InpaintingDataset(Dataset):
    def __init__(self, img, mask, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.mask = mask
        self.img = img
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return 1

    def __getitem__(self, i):
        self.img = self.img.transpose((2, 0, 1))
        result = dict(image=self.img, mask=self.mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        # register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        # device = torch.device(predict_config.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        print(device)
        print(len(dataset), predict_config.dataset)
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)
    
    return cur_res

# @hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def inpaint(img, mask):    
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    indir = current_path + '/data'
    # outdir = r'D:/project/paper_flatten/code/0119/Document-Image-Flattening/lama/result'
    model_path = current_path + '/big-lama'
    checkpoint = 'best.ckpt'
    dataset_img_suffix = '.png'
    dataset_pad_out_to_modulo = 8
    out_key = 'inpainted'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    # out_ext = predict_config.get('out_ext', '.png')

    checkpoint_path = os.path.join(model_path, 'models', checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    # if not predict_config.get('refine', False):
    model.to(device)

    if not indir.endswith('/'):
        indir += '/'

    dataset = InpaintingDataset(img, mask, img_suffix=dataset_img_suffix, pad_out_to_modulo=dataset_pad_out_to_modulo)
    # dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
    # print(device)
    # print(len(dataset), predict_config.dataset)
    for img_i in tqdm.trange(len(dataset)):
        # mask_fname = dataset.mask_filenames[img_i]
        # cur_out_fname = os.path.join(
        #     predict_config.outdir, 
        #     os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
        # )
        # os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        batch = default_collate([dataset[img_i]])
        # if predict_config.get('refine', False):
        #     assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
        #     # image unpadding is taken care of in the refiner, so that output image
        #     # is same size as the input image
        #     cur_res = refine_predict(batch, model, **predict_config.refiner)
        #     cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
        # else:
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)                    
            cur_res = batch[out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        # cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(cur_out_fname, cur_res)
    
    return cur_res

if __name__ == '__main__':

    start = time.time()
    main()
    print(time.time() - start)
