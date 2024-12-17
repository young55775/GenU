import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
image_size = 512
num_images = 1000
min_brightness = 0.1
max_brightness = 0.7
min_size = 1
max_size = 8

background_base_range = (0.3, random.random()*0.3+0.5)
background_blur_sigma = random.random()*4+6
noise_intensity = 0.1+0.1*random.random()

background_center_bias_std = image_size // 6
halo_sigma = random.random()*20+50

min_block_size = 20
max_block_size = 100
num_blocks = 5

def main(i):
    gradient = np.linspace(0, np.random.uniform(*background_base_range), image_size)
    background = np.tile(gradient, (image_size, 1))
    bg_center_x = int(np.random.normal(image_size // 2, background_center_bias_std))
    bg_center_y = int(np.random.normal(image_size // 2, background_center_bias_std))
    bg_center_x = np.clip(bg_center_x, 0, image_size - 1)
    bg_center_y = np.clip(bg_center_y, 0, image_size - 1)
    y, x = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((x - bg_center_x) ** 2 + (y - bg_center_y) ** 2)
    halo = np.exp(-distance / (image_size // 2)) * np.random.uniform(0.3, 0.7)
    background += gaussian_filter(halo, sigma=halo_sigma)
    noise = np.random.normal(0, noise_intensity, (image_size, image_size))
    background += noise
    background = np.clip(background, 0, 1)

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    image = background.copy()
    radius_info = []
    num_spots = random.randint(300, 500)

    for _ in range(num_spots):
        center_x = np.random.randint(0, image_size)
        center_y = np.random.randint(0, image_size)
        radius = np.random.randint(min_size, max_size)
        brightness = np.random.uniform(min_brightness, max_brightness)
        y, x = np.ogrid[:image_size, :image_size]
        mask_circle = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        spot = np.zeros_like(image)
        spot[mask_circle] = brightness
        blur_sigma = np.random.uniform(0.2, 1)
        spot = gaussian_filter(spot, sigma=blur_sigma)
        image += spot
        mask[mask_circle] = 1
        radius_info.append((center_x, center_y, radius))

    for _ in range(num_blocks):
        block_width = random.randint(min_block_size, max_block_size)
        block_height = random.randint(min_block_size, max_block_size)
        top_left_x = np.random.randint(0, image_size - block_width)
        top_left_y = np.random.randint(0, image_size - block_height)

        image[top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 0

        mask[top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 0

    cv2.imwrite(f'./mask/train/img/{i}.tif', image)
    np.save(f'./mask/train/mask/{i}.npy', mask)
if __name__ == "__main__":
    process_map(main,list(range(num_images)),max_workers=6)