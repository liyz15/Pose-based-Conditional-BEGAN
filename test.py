import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import numpy as np
from utils import util
from math import cos, sin
import os

from dataset import CelebADataset


def plot_model():
    model3d = scipy.io.loadmat('model/model3Ddlib.mat')['model_dlib']['threedee'][0][0]
    fig = plt.figure()
    ax = Axes3D(fig)
    x_pos = model3d[:, 0]
    y_pos = model3d[:, 1]
    z_pos = model3d[:, 2]
    ax.scatter(x_pos, y_pos, z_pos)
    for i in range(x_pos.shape[0]):
        ax.text(x_pos[i], y_pos[i], z_pos[i], str(i))
    plt.show()


def plot_celeba():
    img_path = '/home/liyizhuo/Desktop/dataset/CelebA/samples/000001.jpg'
    landmarks = [165, 184, 244, 176, 196, 249, 194, 271, 266, 260]
    x_pos = landmarks[::2]
    y_pos = landmarks[1::2]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.scatter(x_pos, y_pos)
    for i in range(len(x_pos)):
        plt.text(x_pos[i], y_pos[i], str(i))
    plt.show()


def get_model():
    model3d = scipy.io.loadmat('model/model3Ddlib.mat')['model_dlib']['threedee'][0][0]
    res_model3d = np.zeros((5, 3))
    res_model3d[0, :] = np.mean(model3d[[37, 38, 41, 40], :], axis=0)
    res_model3d[1, :] = np.mean(model3d[[43, 44, 47, 46], :], axis=0)
    res_model3d[2, :] = model3d[30, :]
    res_model3d[3, :] = model3d[48, :]
    res_model3d[4, :] = model3d[54, :]

    out_a = scipy.io.loadmat('model/model3Ddlib.mat')['model_dlib']['outA'][0][0]

    # 交换 1 2 列
    res_model3d[:, [1, 2]] = res_model3d[:, [2, 1]]
    res_model3d[:, 2] = -res_model3d[:, 2]

    # return res_model3d, out_a
    return np.ascontiguousarray(res_model3d, dtype=np.float32), np.ascontiguousarray(out_a, dtype=np.float32)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    # this is where different from hopenet
    roll = -(roll * np.pi / 180)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def read_img(img_name):
    img_dir = '/home/liyizhuo/Desktop/dataset/CelebA/samples'
    bbox_file = '/home/liyizhuo/Desktop/dataset/CelebA/Anno/list_bbox_celeba.txt'
    landmarks_file = '/home/liyizhuo/Desktop/dataset/CelebA/Anno/list_landmarks_align_celeba.txt'
    unaligned_landmarks_file = '/home/liyizhuo/Desktop/dataset/CelebA/Anno/list_landmarks_celeba.txt'
    img_path = os.path.join(img_dir, img_name)
    assert os.path.exists(img_path)
    with open(bbox_file, 'r') as f:
        line = f.readline()
        while line.split(' ')[0] != img_name:
            line = f.readline()
    bbox = [int(x) for x in line.split(' ')[1:] if x != '']
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    with open(landmarks_file, 'r') as f:
        line = f.readline()
        while line.split(' ')[0] != img_name:
            line = f.readline()

    landmarks = [int(x) for x in line.split(' ')[1:] if x != '']

    with open(unaligned_landmarks_file, 'r') as f:
        line = f.readline()
        while line.split(' ')[0] != img_name:
            line = f.readline()

    unaligned_landmarks = [int(x) for x in line.split(' ')[1:] if x != '']
    img = cv2.imread(img_path)

    return img, bbox, landmarks, unaligned_landmarks


def test_algorithm():

    # plot_model()  # Plot 3D model with index provided here: https://github.com/dougsouza/face-frontalization
    # plot_celeba()  # Plot celeba samples with landmark
    # 可知 5 个 landmark 分别对应（正视人脸）左眼，右眼，鼻尖，左嘴角，右嘴角
    # 分别对应模型中的 [37, 38, 41, 40] 平均， [43, 44, 47, 46] 平均，30，48，54 (0-index)
    # img_path = '/home/liyizhuo/Desktop/dataset/CelebA/samples/000002.jpg'
    # bbox = np.array([72, 94, 221, 306])
    # unaligned_landmarks = np.array([140, 204, 220, 204, 168, 254, 146, 289, 226, 289])
    # landmarks = np.array([69, 110, 107, 112,  81, 135,  70, 151, 108, 153], dtype=np.float)

    for i in range(1, 101):
        img_name = '{:06d}.jpg'.format(i)
        img, bbox, landmarks, unaligned_landmarks = read_img(img_name)

        # symmetry right eye to left eye

        landmarks = np.reshape(landmarks, (-1, 2))
        landmarks = landmarks.astype(np.float)

        desired_width = 250
        desired_height = 250
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # print(bbox_height)
        # print(bbox_width)
        # landmarks[:, 0] += (bbox_height - bbox_width) / 2
        # landmarks *= 2
        # landmarks[:, 0] += (bbox_width) / 2
        # landmarks[:, 1] += (bbox_height) / 2
        # landmarks[:, 0] *= (desired_width / bbox_width)
        # landmarks[:, 1] *= (desired_width / bbox_width)
        # landmarks[:, 0] *= (desired_height / bbox_height)
        # landmarks[:, 1] *= (desired_height / bbox_height)

        # 图片应该 resize 到 250 * 250 大小，相应的 landmark 位置也会改变

        model3d, out_a = get_model()
        model3d = model3d.astype(np.float)

        landmarks = landmarks.astype(np.float)

        # size = img.shape
        # center = (size[1], size[0])
        # focal_length = center[0] / np.tan(60/2 * np.pi / 180)
        # out_a = np.array(
        #                      [[focal_length, 0, center[0]],
        #                      [0, focal_length, center[1]],
        #                      [0, 0, 1]], dtype= "double"
        #                      )
        # landmarks[:, 0] = bbox_width - 1 - landmarks[:, 0]
        # landmarks[[0, 1, 2, 3, 4], :] = landmarks[[1, 0, 2, 4, 3], :]

        proj_matrix, camera_matrix, rmat, tvec = util.estimate_camera(model3d, out_a, landmarks)

        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [np.radians(_) for _ in eulerAngles]

        # pitch, yaw, roll = np.pi / 4, 0, 0

        # pitch = -np.degrees(pitch)
        # roll = -np.degrees(roll)
        # yaw = -np.degrees(yaw)
        # print(pitch / np.pi * 180)

        pitch = np.degrees(np.arcsin(np.sin(pitch)))
        roll = -np.degrees(np.arcsin(np.sin(roll)))
        yaw = np.degrees(np.arcsin(np.sin(yaw)))

        # print(pitch, yaw, roll)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_axis(img, yaw, pitch, roll, unaligned_landmarks[4], unaligned_landmarks[5])
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 10)

        height, width, _ = img.shape
        half_len = min(bbox_width, bbox_height) // 2
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        bbox[0], bbox[1], bbox[2], bbox[3] = (center_x - half_len,
                                              center_y - half_len,
                                              center_x + half_len,
                                              center_y + half_len)

        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 10)
        print(bbox)
        print((roll, pitch, yaw))

        for l in landmarks:
            img = cv2.circle(img, tuple(l.astype(np.int)), 5, (0, 255, 0))
        landmarks[:, 0] = bbox_width - 1 - landmarks[:, 0]
        for l in landmarks:
            img = cv2.circle(img, tuple(l.astype(np.int)), 5, (0, 255, 255))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/home/liyizhuo/Desktop/dataset/CelebA/samples/test_{}'.format(img_name), img)
        # img = draw_axis(img, 0, 0, 45, 200, 200)
        # plt.imshow(img)
        # plt.show()

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # x_pos = model3d[:, 0]
        # y_pos = model3d[:, 1]
        # z_pos = model3d[:, 2]
        # ax.scatter(x_pos, y_pos, z_pos)
        # for i in range(x_pos.shape[0]):
        #     ax.text(x_pos[i], y_pos[i], z_pos[i], str(i))
        # plt.show()

        # print(model3d)
        # plot_celeba()


if __name__ == '__main__':
    config = {'data_dir': '/home/liyizhuo/Desktop/dataset/CelebA',
              'batch_size': 32,
              'img_size': 64}
    dataset = CelebADataset(config)
    # test_algorithm()
    print(dataset.poses * 90)
    print(dataset.poses_flip * 90)
    print(dataset.bboxs)
