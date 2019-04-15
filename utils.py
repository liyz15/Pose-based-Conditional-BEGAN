# The code for camera calibration is modified from
# https://github.com/dougsouza/face-frontalization/blob/master/camera_calibration.py

import os
import logging
import numpy as np
import cv2
import scipy.io
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def estimate_camera(model3D, out_A, fidu_XY):
    rmat, tvec = calib_camera(model3D, out_A, fidu_XY)
    RT = np.hstack((rmat, tvec))
    projection_matrix = np.dot(out_A, RT)
    return projection_matrix, out_A, rmat, tvec


def calib_camera(model_TD, out_A, fidu_XY):
    ret, rvecs, tvec = cv2.solvePnP(model_TD, fidu_XY, out_A, None, None, None, False)
    rmat, jacobian = cv2.Rodrigues(rvecs, None)

    return rmat, tvec


class PoseCalculator(object):
    def __init__(self, model_path='model/model3Ddlib.mat'):
        self.model3d, self.out_a = self._get_model(model_path)

    def compute(self, landmarks):
        proj_matrix, camera_matrix, rmat, tvec = estimate_camera(self.model3d, self.out_a, landmarks)
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [np.radians(_) for _ in euler_angles]

        pitch = np.degrees(np.arcsin(np.sin(pitch)))
        roll = -np.degrees(np.arcsin(np.sin(roll)))
        yaw = np.degrees(np.arcsin(np.sin(yaw)))

        # return pitch, roll, yaw
        return roll, pitch, yaw

    @staticmethod
    def _get_model(model_path):
        model3d = scipy.io.loadmat(model_path)['model_dlib']['threedee'][0][0]
        res_model3d = np.zeros((5, 3))
        res_model3d[0, :] = np.mean(model3d[[37, 38, 41, 40], :], axis=0)
        res_model3d[1, :] = np.mean(model3d[[43, 44, 47, 46], :], axis=0)
        res_model3d[2, :] = model3d[30, :]
        res_model3d[3, :] = model3d[48, :]
        res_model3d[4, :] = model3d[54, :]

        out_a = scipy.io.loadmat(model_path)['model_dlib']['outA'][0][0]

        # exchange column 1 and 2
        res_model3d[:, [1, 2]] = res_model3d[:, [2, 1]]
        res_model3d[:, 2] = -res_model3d[:, 2]

        # return res_model3d, out_a
        return np.ascontiguousarray(res_model3d, dtype=np.float32), np.ascontiguousarray(out_a, dtype=np.float32)


def draw_debug_image(batch, image_path):
    """
    Draw and save images for a single batch.

    :param batch:
           Torch batch to draw, shape of (batch_size, 3 or 6, image_size, image_size), the batch_size should be
           a multiple of 8, the batch should be scaled to [-1, 1], the pose in the last 3 channel will be
           ignored.
    :param image_path:
           Path to save the drawn image.
    """
    batch = batch.detach().cpu().numpy()
    batch = np.transpose(batch, (0, 2, 3, 1))
    batch = (batch + 1.0) / 2.0 * 255.0
    batch = np.round(batch).astype(np.int)
    batch[batch < 0] = 0
    batch[batch > 255] = 255
    if batch.shape[3] > 3:
        batch = batch[:, :, :, :3]
    batch = batch.astype(np.uint8)

    fig = plt.figure(figsize=(10, 10))
    for i in range(8):
        for j in range(batch.shape[0] // 8):
            ax = plt.subplot(8, batch.shape[0] // 8, i * (batch.shape[0] // 8) + j + 1)
            ax.set_aspect('equal')
            plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.imshow(batch[i * (batch.shape[0] // 8) + j])

    fig.savefig(image_path)
    plt.close(fig)


def create_logger(logger_name,
                  log_format=None,
                  log_level=logging.INFO,
                  log_path=None):
    logger = logging.getLogger(logger_name)
    assert (len(logger.handlers) == 0)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    if log_format is not None:
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path is not None:
        os.stat(os.path.dirname(os.path.abspath(log_path)))
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        if log_format is not None:
            formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
