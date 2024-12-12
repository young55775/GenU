import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
import cv2
from tqdm import tqdm

image_size = 512
num_images = 1300
min_brightness = 0.1
max_brightness = 0.6
min_size = 1
max_size = 8

background_base_range = (0.3, 0.5)
background_blur_sigma = 10
noise_intensity = 0.1

background_center_bias_std = image_size // 6
halo_sigma = 50

for i in tqdm(range(num_images)):
    gradient = np.linspace(0, np.random.uniform(*background_base_range), image_size)
    background = np.tile(gradient, (image_size, 1))
    bg_center_x = int(np.random.normal(image_size // 2, background_center_bias_std))
    bg_center_y = int(np.random.normal(image_size // 2, background_center_bias_std))
    bg_center_x = np.clip(bg_center_x, 0, image_size - 1)
    bg_center_y = np.clip(bg_center_y, 0, image_size - 1)
    y, x = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((x - bg_center_x) ** 2 + (y - bg_center_y) ** 2)
    halo = np.exp(-distance / (image_size // 2)) * np.random.uniform(0.3, 0.5)
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
        blur_sigma = np.random.uniform(0.5, 2)
        spot = gaussian_filter(spot, sigma=blur_sigma)
        image += spot
        mask[mask_circle] = 1
        radius_info.append((center_x, center_y, radius))
    # image = np.clip(image, 0, 1)
    #
    # image_uint8 = (image * 100000).astype(np.uint8)
    cv2.imwrite(f'./mask_data/{i}.tif', image)
    np.save(f'./mask_data/{i}.npy', mask)
