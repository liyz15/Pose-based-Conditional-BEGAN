import os
import time

import numpy as np
import cv2

from utils import util

np.random.seed(233)


class CelebADataset(object):
    """Dataset for CelebA"""
    def __init__(self, config=None):
        start = time.time()
        self.config = config
        self.batch_size = config.getint('batch_size')
        self.img_size = config.getint('image_size', 64)
        assert os.path.exists(config['data_dir']), 'data dir {} does not exist'.format(config['data_dir'])

        # read data
        self.pose_calculator = util.PoseCalculator()
        self.filenames, self.bboxs, self.poses, self.poses_flip = self._parse_dataset()

        self.samples_num = len(self.filenames)
        self.index_seq = np.arange(0, self.samples_num)
        self.current_index = 0
        # np.random.shuffle(self.index_seq)

        print('Time to build dataset: {:.2f}s'.format(time.time() - start))

    def generate_batch(self):
        # Not enough samples left, shuffle the index and recount
        if self.current_index + self.batch_size >= self.samples_num:
            self.current_index = 0
            np.random.shuffle(self.index_seq)

        batch_index = self.index_seq[self.current_index:(self.current_index + self.batch_size)]
        self.current_index = self.current_index + self.batch_size

        batch = np.zeros((self.batch_size, 3, self.img_size, self.img_size))
        poses = np.zeros((self.batch_size, 3))

        for i in range(self.batch_size):
            bbox = self.bboxs[batch_index[i]]
            img_path = os.path.join(self.config['data_dir'], 'img_celeba', self.filenames[batch_index[i]])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            poses[i] = self.poses[batch_index[i]]

            # Flip
            if self.config.getboolean('flip', True) and np.random.rand() > 0.5:
                img = np.flip(img, 1)
                poses[i] = self.poses_flip[batch_index[i]]

            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            # Convert from (H, W, C) to (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            # Scale to [-1, 1]
            img = (img / 255.0 * 2) - 1.0
            batch[i] = img

        return batch, poses

    def _parse_dataset(self):
        """
        :return:
            filenames: list of filenames
            bboxs: Bounding box of shape (N, 4), (x_min, y_min, x_max, y_max)
            poses: Poses of shape (N, 3), (roll, pitch, yaw)
            poses_flip: Flipped Poses of shape (N, 3), (roll, pitch, yaw)
        """
        # Read filenames and bboxs
        bbox_file = os.path.join(self.config['data_dir'], 'Anno', 'list_bbox_celeba.txt')
        assert os.path.exists(bbox_file)
        bboxs = open(bbox_file).readlines()
        filenames = [x.split()[0] for x in bboxs[2:]]
        bboxs = [[int(y) for y in x.split()[1:]] for x in bboxs[2:]]
        bboxs = np.array(bboxs)

        wh = np.zeros((bboxs.shape[0], 2), dtype=np.int)
        wh[:, 0] = bboxs[:, 2]
        wh[:, 1] = bboxs[:, 3]
        half_len = np.min(wh, axis=1) // 2

        # Add height and width
        bboxs[:, 2] = bboxs[:, 0] + wh[:, 0]
        bboxs[:, 3] = bboxs[:, 1] + wh[:, 1]

        # center crop to square
        assert half_len.shape[0] == bboxs.shape[0]
        center = np.zeros_like(wh)
        center[:, 0] = (bboxs[:, 0] + bboxs[:, 2]) / 2
        center[:, 1] = (bboxs[:, 1] + bboxs[:, 3]) / 2
        bboxs[:, 0], bboxs[:, 1], bboxs[:, 2], bboxs[:, 3] = (center[:, 0] - half_len,
                                                              center[:, 1] - half_len,
                                                              center[:, 0] + half_len,
                                                              center[:, 1] + half_len)

        # Read landmark and compute pose
        landmark_file = os.path.join(self.config['data_dir'], 'Anno', 'list_landmarks_align_celeba.txt')
        assert os.path.exists(landmark_file)
        landmarks = open(landmark_file).readlines()
        landmarks = np.array([[int(y) for y in x.split()[1:]] for x in landmarks[2:]], dtype=np.float)
        landmarks = np.reshape(landmarks, (-1, 5, 2))
        poses = [self.pose_calculator.compute(x) for x in landmarks]
        poses = np.squeeze(np.array(poses))
        poses /= 90.0

        # Flip
        landmarks[:, :, 0] = wh[:, 0, None] - 1 - landmarks[:, :, 0]
        landmarks[:, [0, 1, 2, 3, 4], :] = landmarks[:, [1, 0, 2, 4, 3], :]
        poses_flip = [self.pose_calculator.compute(x) for x in landmarks]
        poses_flip = np.squeeze(np.array(poses_flip))
        poses_flip /= 90.0

        # Check shape
        print('Number of images: {}'.format(len(filenames)))
        print('Bounding boxes shape: {}'.format(bboxs.shape))
        print('Poses shape: {}'.format(poses.shape))
        print('Poses flip shape: {}'.format(poses.shape))
        assert bboxs.shape[0] == poses.shape[0], 'Shape does not match'

        return filenames, bboxs, poses, poses_flip
