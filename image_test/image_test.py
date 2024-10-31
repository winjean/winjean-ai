from load_model import portrait_matting, universal_matting
from modelscope.outputs import OutputKeys
from skimage import io, transform
import cv2
import numpy as np


def remove_background(input_img, background_img):
    result = portrait_matting(input_img)
    # result = universal_matting(input_img)
    temp_image = r'../image/temp_image.png'
    cv2.imwrite(temp_image, result[OutputKeys.OUTPUT_IMG])

    res_img = io.imread(temp_image, plugin='pil')  # 读取带有透明通道的图像
    background_img = io.imread(background_img, plugin='pil')  # 读取背景图像
    w, h = res_img.shape[1], res_img.shape[0]
    background_img = transform.resize(background_img, (h, w), mode='reflect', preserve_range=True).astype(np.uint8)
    alpha = res_img[:, :, 3] / 255.0
    input_img_rgb = res_img[:, :, :3]  # 只取 RGB 通道
    combine_img = input_img_rgb * alpha[:, :, np.newaxis] + background_img * (1 - alpha[:, :, np.newaxis])
    combine_img = combine_img.astype(np.uint8)
    io.imsave(r'../image/output.png', combine_img)


if __name__ == '__main__':
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    remove_background(r'../image/portrait.png', r'../image/background.jpg')
