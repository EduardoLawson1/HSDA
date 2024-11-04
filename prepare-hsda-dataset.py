import argparse
import numpy as np
import random
import os
from PIL import Image


def low_pass_gaussian_filter(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = np.exp(-1 * dis_square / (2 * D ** 2))
    return template * fshift


def high_pass_gaussian_filter(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - np.exp(-1 * dis_square / (2 * D ** 2))
    return template * fshift


def ifft(fshift):
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifftn(ishift)
    iimg = np.abs(iimg)
    return iimg


def f_shift(img_c):  # given one channel image
    f = np.fft.fftn(img_c)
    f_shift = np.fft.fftshift(f)
    return f_shift


def f_shift_rgb(img):
    img = np.array(img)
    f_shift_r = f_shift(img[:, :, 0])
    f_shift_g = f_shift(img[:, :, 1])
    f_shift_b = f_shift(img[:, :, 2])
    f_shift_rgb = np.array([f_shift_r, f_shift_g, f_shift_b])
    return f_shift_rgb


def low_pass_rgb(img, D):
    img_f_shift_rgb = f_shift_rgb(img)
    low_part_r = low_pass_gaussian_filter(img_f_shift_rgb[0].copy(), D)
    low_part_g = low_pass_gaussian_filter(img_f_shift_rgb[1].copy(), D)
    low_part_b = low_pass_gaussian_filter(img_f_shift_rgb[2].copy(), D)
    low_rgb = np.array([low_part_r, low_part_g, low_part_b])
    return low_rgb


def high_pass_rgb(img, D):
    img_f_shift_rgb = f_shift_rgb(img)
    high_part_r = high_pass_gaussian_filter(img_f_shift_rgb[0].copy(), D)
    high_part_g = high_pass_gaussian_filter(img_f_shift_rgb[1].copy(), D)
    high_part_b = high_pass_gaussian_filter(img_f_shift_rgb[2].copy(), D)
    high_rgb = np.array([high_part_r, high_part_g, high_part_b])
    return high_rgb


def low_high_pass_rgb(img, D):
    low_rgb = low_pass_rgb(img, D)
    high_rgb = high_pass_rgb(img, D)
    low_high_rgb = np.array([low_rgb, high_rgb])
    return low_high_rgb


def shuffle_image(img, D, k):
    channel_num = random.randrange(0, 3)
    img_low_high_parts = low_high_pass_rgb(img, D)

    target = img_low_high_parts[1][channel_num]  # spectrum to shuffle
    magnitudes = np.abs(target)  # 900 by 1600 for nuscenes cam image
    target_indices_flattened = np.argpartition(magnitudes.flatten(),
                                               -1 * k)[-1 * k:]
    target_indices = [np.unravel_index(i, magnitudes.shape)
                      for i in target_indices_flattened]

    new_indices = random.sample(target_indices, len(target_indices))
    for i, pixel in enumerate(target_indices):
        new = new_indices[i]  # pixel to swap with
        new_y = new[0]
        new_x = new[1]
        old_y = pixel[0]
        old_x = pixel[1]
        target[old_y, old_x], target[new_y, new_x] = (target[new_y, new_x],
                                                      target[old_y, old_x])

    shuffled_low_high_parts_r = (img_low_high_parts[0][0] +
                                 img_low_high_parts[1][0])
    shuffled_low_high_parts_g = (img_low_high_parts[0][1] +
                                 img_low_high_parts[1][1])
    shuffled_low_high_parts_b = (img_low_high_parts[0][2] +
                                 img_low_high_parts[1][2])

    img_r = ifft(shuffled_low_high_parts_r)
    img_g = ifft(shuffled_low_high_parts_g)
    img_b = ifft(shuffled_low_high_parts_b)
    h, w = shuffled_low_high_parts_r.shape  # any channel is fine
    rgbArray = np.zeros((h, w, 3), 'uint8')
    rgbArray[:, :, 0] = img_r
    rgbArray[:, :, 1] = img_g
    rgbArray[:, :, 2] = img_b
    img = Image.fromarray(rgbArray)
    return img


def main():
    parser = argparse.ArgumentParser(description='HSDA')
    parser.add_argument('--d', type=int, default=10,
                        help='determines delineation of high and low \
                              frequencies')
    parser.add_argument('--k', type=int, default=2000,
                        help='how many of the top frequencies to shuffle \
                              in the high frequency spectrum')
    args = parser.parse_args()

    # Prepare new HSDA dataset.
    # Symlink to avoid wasting space.
    assert os.path.isdir('data/nuscenes')
    assert os.path.isdir('data/nuscenes/samples')

    if not os.path.isdir('data/nuscenes-hsda'):
        os.makedirs('data/nuscenes-hsda/')
    if not os.path.isdir('data/nuscenes-hsda/samples'):
        os.makedirs('data/nuscenes-hsda/samples')
    root_dirs_to_symlink = ['maps', 'sweeps', 'v1.0-trainval']
    samples_dirs_to_symlink = ['LIDAR_TOP', 'RADAR_BACK_LEFT',
                               'RADAR_BACK_RIGHT', 'RADAR_FRONT',
                               'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
    for dir in root_dirs_to_symlink:
        if not os.path.isdir('data/nuscenes-hsda/' + dir):
            os.symlink('../nuscenes/' + dir,
                       'data/nuscenes-hsda/' + dir,
                       target_is_directory=True)
    for dir in samples_dirs_to_symlink:
        if not os.path.isdir('data/nuscenes-hsda/samples/' + dir):
            os.symlink('../../nuscenes/samples/' + dir,
                       'data/nuscenes-hsda/samples/' + dir,
                       target_is_directory=True)

    # Generate HSDA images to use instead of regular camera images.
    samples_dir = 'data/nuscenes/samples'
    save_dir = 'data/nuscenes-hsda/samples'

    for dir in os.listdir(samples_dir):
        if dir[0:3] == 'CAM':  # Only process camera images
            print(f'processing {dir}')
            if not os.path.exists(save_dir + '/' + dir):
                os.makedirs(save_dir + '/' + dir)
            dir_full_path = samples_dir + '/' + dir
            for image_filename in os.listdir(dir_full_path):
                save_destination = save_dir + '/' + dir + '/' + image_filename
                if not os.path.isfile(save_destination):
                    print(f'processing {image_filename}')
                    image = Image.open(dir_full_path + '/' + image_filename)
                    shuffled_image = shuffle_image(image, args.d, args.k)
                    shuffled_image.save(save_destination)
                else:
                    print(f'already processed {image_filename}, skipping')


if __name__ == '__main__':
    main()
