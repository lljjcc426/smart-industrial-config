import os
import pygame
import random
from PIL import Image, ImageEnhance, ImageFilter  # 新增 ImageFilter
import numpy as np

pygame.init()

# 配置同模拟器
COLOR_BG = (10, 15, 20)
COLOR_TEXT_GLOW = (0, 255, 0)
COLOR_ALERT = (255, 50, 50)
FONT_NAME = "consolas"
font_name = pygame.font.match_font(FONT_NAME) or pygame.font.get_default_font()
FONT_VAL = pygame.font.Font(font_name, 100)
LCD_WIDTH, LCD_HEIGHT = 240, 120
BORDER_COLOR = (100, 100, 100)

DATASET_DIR = "dataset_lcd"
os.makedirs(DATASET_DIR, exist_ok=True)

def generate_single_lcd(value):
    surface = pygame.Surface((LCD_WIDTH, LCD_HEIGHT))
    surface.fill((0, 0, 0))
    pygame.draw.rect(surface, BORDER_COLOR, (0, 0, LCD_WIDTH, LCD_HEIGHT), 2)
    
    alert = value > 120
    color = COLOR_ALERT if alert else COLOR_TEXT_GLOW
    
    text = FONT_VAL.render(f"{value:03d}", True, color)
    rect = text.get_rect(center=(LCD_WIDTH // 2, LCD_HEIGHT // 2))
    
    # 轻微偏移 + 噪声
    offset_x, offset_y = random.randint(-3, 3), random.randint(-3, 3)
    surface.blit(text, (rect.x + offset_x, rect.y + offset_y))
    
    img_array = pygame.surfarray.array3d(surface).swapaxes(0, 1)
    noise = np.random.normal(0, random.uniform(0.5, 4), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    
    # 随机模糊
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    
    # 随机亮度调整
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))
    
    return img

def generate_data(samples_per_class=500):
    print(f"Generating dataset (000-150, {samples_per_class} samples per class)...")
    for val in range(151):  # 0-150
        class_dir = os.path.join(DATASET_DIR, f"{val:03d}")
        os.makedirs(class_dir, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_single_lcd(val)
            img.save(os.path.join(class_dir, f"{i}.png"))
    print("Dataset generation completed!")

if __name__ == "__main__":
    generate_data()