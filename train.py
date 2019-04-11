import os
import argparse
import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from dataset import CelebADataset
from model import Generator, Discriminator
from utils import util

parser = argparse.ArgumentParser()
np.random.seed(233)
torch.manual_seed(233)


def to_torch(x):
    """Put data, model on the proper device"""
    x = torch.from_numpy(x).float()
    if torch.cuda.is_available():
        x.cuda()
    return x


def generate_z(z_dim, batch_size):
    """Generate a tensor of size (batch_size, z_dim) in range [-1, 1)"""
    return torch.rand((batch_size, z_dim)) * 2.0 - 1.0


def weights_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform(m.weight)
        m.bais.data.fill_(0.01)


def get_target_pose():
    target_pose = torch.zeros((64, 3))
    target_pose[0*8:1*8, 2] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[1*8:2*8, 2] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[2*8:3*8, 2] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[3*8:4*8, 2] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[4*8:5*8, 0] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[5*8:6*8, 0] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[6*8:7*8, 1] = torch.arange(-1.0, 1.1, 2.0 / 7)
    target_pose[7*8:8*8, 1] = torch.arange(-1.0, 1.1, 2.0 / 7)
    if torch.cuda.is_available():
        target_pose.cuda()
    return target_pose


def draw_debug_image(batch, image_path):
    batch = batch.detach().cpu().numpy()
    batch = np.transpose(batch, (0, 2, 3, 1))
    batch = (batch + 1.0) / 2.0 * 255.0
    batch = np.round(batch).astype(np.int)
    batch[batch < 0] = 0
    batch[batch > 255] = 255

    fig = plt.figure(figsize=(10, 10))
    for i in range(8):
        for j in range(batch.shape[0] // 8):
            ax = plt.subplot(8, batch.shape[0], i * 8 + j + 1)
            ax.set_aspect('equal')
            plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.imshow(batch[i * 8 + j])

    fig.savefig(image_path)


def train(config):
    # ******************************************************************************************************************
    # * Build logger
    # ******************************************************************************************************************

    time_now = datetime.datetime.now().strftime('%m-%d-%H%M%S')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = util.create_logger(
        logger_name='main_logger',
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_path='logs/train_{}.log'.format(time_now)
    )

    log_list = {'real_loss_D': [], 'fake_loss_D': [], 'loss_G': []}

    # ******************************************************************************************************************
    # * Build dataset
    # ******************************************************************************************************************
    # Generate fixed input for debugging

    fixed_z = generate_z(config.z_dim, config.batch_size)
    dataset = CelebADataset(config)
    fixed_real_batch, fixed_pose_batch = dataset.generate_batch()
    dataset.current_index = 0  # Reset dataset index

    if torch.cuda.is_available():
        fixed_z.cuda()
    fixed_real_batch = to_torch(fixed_real_batch)
    fixed_pose_batch = to_torch(fixed_pose_batch)
    fixed_real_batch_pose = torch.cat((
        fixed_real_batch,
        fixed_pose_batch.view(-1, 1, 1, 3).expand(-1, config.image_size, config.image_size, -1)), dim=1)
    fixed_z_pose = torch.cat((fixed_z, fixed_pose_batch), dim=1)

    fixed_target_pose = get_target_pose()
    fixed_z_pose_vary = torch.cat((
        fixed_z[:8].view(8, 1, config.z_dim).expand(-1, 8, -1).contiguous().view(64, config.z_dim),
        fixed_target_pose.view(-1, 1, 1, 3).expand(-1, 64, 64, -1)), dim=1)

    # ******************************************************************************************************************
    # * Build model and optimizer
    # ******************************************************************************************************************

    D = Discriminator()
    G = Generator(input_dim=(config.z_dim + 3), output_channel=3)
    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    D.apply(weights_init)
    G.apply(weights_init)

    optimizerD = torch.optim.Adam(D.parameters(), betas=(0.9, 0.99), lr=config.lr)
    optimizerG = torch.optim.Adam(G.parameters(), betas=(0.9, 0.99), lr=config.lr)

    # ******************************************************************************************************************
    # * Train!
    # ******************************************************************************************************************
    lambda_k = config.lambda_k
    step_num = (config.epoch * dataset.samples_num) // config.batch_size
    for step in range(step_num):
        real_batch, pose_batch = dataset.generate_batch()
        real_batch = to_torch(real_batch)
        pose_batch = to_torch(pose_batch)
        z = generate_z(config.z_dim, config.batch_size)
        if torch.cuda.is_available():
            z.cuda()
        z_pose = torch.cat((z, pose_batch), dim=1)

        # Train the discriminator
        D.zero_grad()
        fake_batch = G(z_pose)
        real_batch_pose = torch.cat((
            real_batch,
            pose_batch.view(-1, 1, 1, 3).expand(-1, config.image_size, config.image_size, -1)), dim=1)
        fake_batch_pose = torch.cat((
            fake_batch,
            pose_batch.view(-1, 1, 1, 3).expand(-1, config.image_size, config.image_size, -1)), dim=1)

        real_output_D = D(real_batch_pose)
        fake_output_D = D(fake_batch_pose.detach())

        real_loss_D = torch.mean(torch.abs(real_output_D - real_batch_pose))
        fake_loss_D = torch.mean(torch.abs(fake_output_D - fake_batch_pose))
        loss_D = real_loss_D - lambda_k * fake_loss_D

        loss_D.backward()
        optimizerD.step()

        log_list['real_loss_D'].append(real_loss_D.item())
        log_list['fake_loss_D'].append(fake_loss_D.item())

        # Train the generator
        G.zero_grad()
        fake_batch = G(z_pose)
        fake_batch_pose = torch.cat((
            fake_batch,
            pose_batch.view(-1, 1, 1, 3).expand(-1, config.image_size, config.image_size, -1)), dim=1)
        fake_output_G = D(fake_batch_pose)

        loss_G = torch.mean(torch.abs(fake_output_G - fake_batch_pose))

        loss_G.backward()
        optimizerD.step()

        log_list['loss_G'].append(loss_G.item())

        # Log and Save
        if step % config.verbose_steps == 0:
            logger.info('It: {}\treal_loss_D: {:.4f}\tfake_loss_D: {:.4f}\tloss_G: {:.4f}'.format(
                step + 1,
                np.mean(log_list['real_loss_D']),
                np.mean(log_list['fake_loss_D']),
                np.mean(log_list['loss_G'])
            ))
            for k in log_list.keys():
                log_list[k] = []

        if step % config.verbose_steps == 0:
            checkpoint_dir = 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            torch.save(G.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_{}'.format(step)))
            D.eval()
            G.eval()
            fixed_real_output_D = D(fixed_real_batch_pose)
            fixed_fake_batch = G(fixed_z_pose)
            fixed_fake_batch_vary = G(fixed_z_pose_vary)
            draw_debug_image(fixed_real_batch, os.path.join(checkpoint_dir, 'debug_real_{}'.format(step)))
            draw_debug_image(fixed_real_output_D, os.path.join(checkpoint_dir, 'debug_real_D_{}'.format(step)))
            draw_debug_image(fixed_fake_batch, os.path.join(checkpoint_dir, 'debug_fake_{}'.format(step)))
            draw_debug_image(fixed_fake_batch_vary, os.path.join(checkpoint_dir, 'debug_fake_fixed_{}'.format(step)))
            D.train()
            G.train()

    torch.save(G.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_final'))
    D.eval()
    G.eval()
    fixed_real_output_D = D(fixed_real_batch_pose)
    fixed_fake_batch = G(fixed_z_pose)
    fixed_fake_batch_vary = G(fixed_z_pose_vary)
    draw_debug_image(fixed_real_batch, os.path.join(checkpoint_dir, 'debug_real_{}'.format(step)))
    draw_debug_image(fixed_real_output_D, os.path.join(checkpoint_dir, 'debug_real_D_{}'.format(step)))
    draw_debug_image(fixed_fake_batch, os.path.join(checkpoint_dir, 'debug_fake_{}'.format(step)))
    draw_debug_image(fixed_fake_batch_vary, os.path.join(checkpoint_dir, 'debug_fake_fixed_{}'.format(step)))


if __name__ == '__main__':
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--flip', type=bool, default=True)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lambda_k', type=float, default=1e-3)
    parser.add_argument('verbose_steps', type=int, default=100)
    parser.add_argument('save_steps', type=int, default=5000)

    config = parser.parse_args()
    train(config)
