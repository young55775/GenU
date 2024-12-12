import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_labels(image_path, label_path):
    # 读取图像和标签文件
    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        annotations = f.readlines()

    # 绘制图像和框
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_size
        y_center = float(parts[2]) * image_size
        width = float(parts[3]) * image_size
        height = float(parts[4]) * image_size

        # 绘制框 (bounding box)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 检查框的坐标是否在图像尺寸范围内，防止绘制超出图像的框
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_size - 1, x2)
        y2 = min(image_size - 1, y2)

        # 绘制绿色的矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

        # 绘制关键点
        keypoints = parts[5:]
        for i in range(0, len(keypoints), 2):
            key_x = int(float(keypoints[i]) * image_size)
            key_y = int(float(keypoints[i + 1]) * image_size)
            cv2.circle(image, (key_x, key_y), 1, (0, 0, 255), -1)  # 红色关键点

    # 使用matplotlib显示结果
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # 不显示坐标轴
    plt.show(block=True)

# 用于检查标签和生成图像
image_size = 512
num_images = 100
min_brightness = 0.1
max_brightness = 0.6
min_size = 1
max_size = 10
num_keypoints = 16

for i in tqdm(range(num_images)):
    gradient = np.linspace(0, np.random.uniform(0.3, 0.5), image_size)
    background = np.tile(gradient, (image_size, 1))
    bg_center_x = int(np.random.normal(image_size // 2, image_size // 6))
    bg_center_y = int(np.random.normal(image_size // 2, image_size // 6))
    bg_center_x = np.clip(bg_center_x, 0, image_size - 1)
    bg_center_y = np.clip(bg_center_y, 0, image_size - 1)

    y, x = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((x - bg_center_x) ** 2 + (y - bg_center_y) ** 2)
    halo = np.exp(-distance / (image_size // 2)) * np.random.uniform(0.3, 0.5)
    background += gaussian_filter(halo, sigma=50)
    noise = np.random.normal(0, 0.1, (image_size, image_size))
    background += noise
    background = np.clip(background, 0, 1)

    image = background.copy()
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    yolo_annotations = []

    num_spots = random.randint(50, 120)
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

        bbox_width = 2 * radius
        bbox_height = 2 * radius

        x_center_normalized = center_x / image_size
        y_center_normalized = center_y / image_size
        width_normalized = bbox_width / image_size
        height_normalized = bbox_height / image_size

        keypoints = []
        for j in range(num_keypoints):
            angle = 2 * np.pi * j / num_keypoints
            key_x = center_x + radius * np.cos(angle)
            key_y = center_y + radius * np.sin(angle)
            keypoints.append((key_x, key_y))

        keypoints_normalized = [(kp[0] / image_size, kp[1] / image_size) for kp in keypoints]
        keypoints_str = " ".join([f"{kp[0]} {kp[1]}" for kp in keypoints_normalized])

        yolo_annotations.append(
            f"0 {keypoints_str}")

    cv2.imwrite(f'./datasets/val/{1001+i}.jpg', image * 255)
    with open(f'./datasets/val/{1001+i}.txt', 'w') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')

    # 可视化图像和标签
    #visualize_labels(f'./datasets/val/{1001 + i}.jpg', f'./datasets/val/{1001 + i}.txt')