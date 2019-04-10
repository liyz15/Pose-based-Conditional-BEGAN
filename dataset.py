import os
import time

import numpy as np

from utils import util

np.random.seed(233)


class CelebADataset(object):
    """Dataset for CelebA"""
    def __init__(self, config=None):
        start = time.time()
        self.config = config
        assert os.path.exists(config['data_dir']), 'data dir {} does not exist'.format(config['data_dir'])

        self.pose_calculator = util.PoseCalculator()
        self.filenames, self.bboxs, self.poses, self.poses_flip = self._parse_dataset()

        print('Time to build dataset: {:.2f}s'.format(time.time() - start))

    def _parse_dataset(self):
        """
        :return:
            bboxs: Bounding box of shape (N, 4), (x_min, y_min, x_max, y_max)
            poses: Poses of shape (N, 3), (roll, pitch, yaw)
        """
        # Read filenames and bboxs
        bbox_file = os.path.join(self.config['data_dir'], 'Anno', 'list_bbox_celeba.txt')
        assert os.path.exists(bbox_file)
        bboxs = open(bbox_file).readlines()[:8]
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
        landmarks = open(landmark_file).readlines()[:8]
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
