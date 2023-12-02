import argparse
import os
import os.path as osp
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict

sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger

def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'SUBCONFIGS' in cfg:
        if model_name is not None and model_name in cfg['SUBCONFIGS']:
            cfg.update(cfg['SUBCONFIGS'][model_name])
        del cfg['SUBCONFIGS']

    return edict(cfg) if return_edict else cfg

def main():
    args, cfg = parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    net.set_resolution(args.hr_h, args.hr_w, args.lr, False)
    net.is_sim = args.is_sim
    predictor = Predictor(net, device)

    process_single_image(args.image, args.mask, predictor, cfg, args)

def process_single_image(image_path, mask_path, predictor, cfg, args):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = image.shape
    image = cv2.resize(image, (args.hr_h, args.hr_w), cv2.INTER_LINEAR)

    mask_image = cv2.imread(mask_path).astype(np.float32) / 255
    mask_image = cv2.resize(mask_image, (args.hr_h, args.hr_w), cv2.INTER_LINEAR)
    mask = mask_image[:, :, 0]

    pred = predictor.predict(image, mask, return_numpy=False)
    pred = pred.detach().cpu().numpy().astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    if args.original_size:
        pred = cv2.resize(pred, image_size[:-1][::-1])

    image_name = os.path.basename(image_path)
    _save_image(image_name, pred, cfg.RESULTS_PATH)

def _save_image(image_name, bgr_image, results_path):
    # 确保保存路径存在
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # 构建完整的文件保存路径
    save_path = os.path.join(results_path, image_name)

    # 将图像保存到指定路径
    cv2.imwrite(save_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # 打印消息确认图像已保存
    print(f"Image saved as {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    
    parser.add_argument('--image', type=str, help='Path to a single image file.')
    parser.add_argument('--mask', type=str, help='Path to a single mask file.')

    parser.add_argument('--lr', type=int, default=256, help='base resolution')
    parser.add_argument('--hr_h', type=int, default=1024, help='target h resolution')
    parser.add_argument('--hr_w', type=int, default=1024, help='target w resolution')
    parser.add_argument('--is_sim', action='store_true', default=False,
                        help='Whether use CDTNet-sim.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument(
        '--original-size', action='store_true', default=False,
        help='Resize predicted image back to the original size.'
    )
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml', help='The path to the config file.')
    parser.add_argument(
        '--results-path', type=str, default='',
        help='The path to the harmonized images. Default path: cfg.EXPS_PATH/predictions.'
    )

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
    cfg.RESULTS_PATH = Path(args.results_path) if len(args.results_path) else cfg.EXPS_PATH / 'predictions'
    cfg.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(cfg)
    return args, cfg

# 封装成函数

def process_single_image(model_type, checkpoint, image_path, mask_path, results_path, hr_h=2048, hr_w=2048, lr=512, gpu=0, is_sim=False, original_size=False):
    os.chdir('/data3/chenjh/yxj_eval_task/CDTNet-single')
    device = torch.device(f'cuda:{gpu}')
    # 构建模型检查点的完整路径
    checkpoint_path = os.path.join('./CDTNet-single', checkpoint)

    # 加载模型
    net = load_model(model_type, checkpoint_path, verbose=True)
    net.set_resolution(hr_h, hr_w, lr, False)
    net.is_sim = is_sim
    predictor = Predictor(net, device)


    # 处理单张图像和掩码
    # 读取图像和掩码
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = image.shape
    image = cv2.resize(image, (hr_h, hr_w), cv2.INTER_LINEAR)

    mask_image = cv2.imread(mask_path).astype(np.float32) / 255
    mask_image = cv2.resize(mask_image, (hr_h, hr_w), cv2.INTER_LINEAR)
    mask = mask_image[:, :, 0]

    # 使用模型进行预测
    pred = predictor.predict(image, mask, return_numpy=False)
    pred = pred.detach().cpu().numpy().astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    # 调整大小
    if original_size:
        pred = cv2.resize(pred, image_size[:-1][::-1])

    # 保存图像
    image_name = os.path.basename(image_path)
    results_path = Path(results_path)
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    save_path = results_path / image_name
    cv2.imwrite(str(save_path), pred, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Image saved as {save_path}")

    os.chdir('/data3/chenjh/yxj_eval_task')

if __name__ == '__main__':
    main()