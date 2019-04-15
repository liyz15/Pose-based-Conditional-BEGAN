import os
import argparse

import torch
import torch.nn as nn
import numpy as np

from model import Generator
from utils import draw_debug_image
from train import get_target_pose, generate_z

parser = argparse.ArgumentParser()


def inference(config):
    assert os.path.exists(config.checkpoint), 'The checkpoint {} doesn\'t exists'.format(config.checkpoint)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if config.random_seed is not None:
        np.random.seed(config.random_sedd)
        torch.random.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.seed_all(config.random_seed)

    print('Build up model.')
    G = Generator(input_dim=(config.z_dim + 3), output_channel=3)
    print('Load generator from {}.'.format(config.checkpoint))
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(config.checkpoint))
    else:
        G.load_state_dict(torch.load(config.checkpoint, map_location='cpu'))

    z = torch.rand((64, config.z_dim)) * 2.0 - 1.0
    # pose_batch = torch.rand((64, 3)) * 0.5 - 0.25
    pose_batch = torch.zeros((64, 3))

    if torch.cuda.is_available():
        z = z.cuda()
        pose_batch = pose_batch.cuda()
        G.cuda()

    z_pose = torch.cat((z, pose_batch), dim=1)

    # Get a target batch
    target_pose_batch = get_target_pose()
    fixed_z_pose = torch.cat((
        z[:8].view(8, 1, config.z_dim).expand(-1, 8, -1).contiguous().view(64, config.z_dim),
        target_pose_batch), dim=1)

    output = G(z_pose)
    draw_debug_image(output, os.path.join(config.output_dir, 'output.png'))
    print('Image saved at: {}'.format(os.path.join(config.output_dir, 'output.png')))
    output = G(fixed_z_pose)
    draw_debug_image(output, os.path.join(config.output_dir, 'output_fix.png'))
    print('Image saved at: {}'.format(os.path.join(config.output_dir, 'output_fix.png')))


if __name__ == '__main__':
    parser.add_argument('--checkpoint', type=str, required=True, help='Path of the checkpoint of generator.')
    parser.add_argument('--output_dir', type=str, default='output', help='Output dir for saving generated images.')
    parser.add_argument('--z_dim', type=int, default=128, help='Length of noise for the generator.')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for numpy and torch.')

    config = parser.parse_args()
    inference(config)
