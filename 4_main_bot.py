import time
import cv2
import numpy as np
import mss
import pyautogui
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# 模型定义
class ScadaLCDNet(nn.Module):
    def __init__(self, num_classes=151):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# 配置
MODEL_PATH = "model_out/scada_lcd_net_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_VIEW = True 

transform = transforms.Compose([
    transforms.Resize((120, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        sys.exit(1)
    model = ScadaLCDNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

def predict(model, roi_pil):
    img_tensor = transform(roi_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        return torch.argmax(output, 1).item()

last_alert = False

def main_loop():
    global last_alert
    model = load_model()
    sct = mss.mss()
    
    monitor_info = sct.monitors[1]
    # 缩放修正
    scale_x = pyautogui.size()[0] / monitor_info['width']
    scale_y = pyautogui.size()[1] / monitor_info['height']
    
    print(f">>> 智能相对定位版启动")

    while True:
        try:
            screenshot = np.array(sct.grab(monitor_info))
            img_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
            cnts_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            btn_box = None
            for cnt in sorted(cnts_blue, key=cv2.contourArea, reverse=True):
                if 2000 < cv2.contourArea(cnt) < 50000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if 2.0 < w/h < 6.0:
                        btn_box = (x, y, w, h)
                        break
            
            if btn_box:
                bx, by, bw, bh = btn_box

                search_x1 = max(0, bx - 50)
                search_x2 = min(img_bgr.shape[1], bx + bw + 50)
                search_y1 = max(0, by - 500)
                search_y2 = max(0, by - 20)

                roi_search = img_bgr[search_y1:search_y2, search_x1:search_x2]
                
                if roi_search.size > 0:
                    roi_hsv = cv2.cvtColor(roi_search, cv2.COLOR_BGR2HSV)
                    mask_black = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
                    cnts_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    lcd_box = None
                    for cnt in sorted(cnts_black, key=cv2.contourArea, reverse=True):
                        if cv2.contourArea(cnt) > 5000:
                            lx, ly, lw, lh = cv2.boundingRect(cnt)
                            # 检查比例
                            if 1.5 < lw/lh < 3.0:
                                # 坐标转换
                                lcd_box = (search_x1 + lx, search_y1 + ly, lw, lh)
                                break
                    
                    if lcd_box:
                        lx, ly, lw, lh = lcd_box
                        
                        # 截取真正的 LCD 图片
                        roi_lcd = img_bgr[ly:ly+lh, lx:lx+lw]
                        roi_pil = Image.fromarray(cv2.cvtColor(roi_lcd, cv2.COLOR_BGR2RGB))
                        
                        # 识别
                        val = predict(model, roi_pil)
                        
                        # 计算点击坐标
                        click_x = int((monitor_info["left"] + bx + bw//2) * scale_x)
                        click_y = int((monitor_info["top"] + by + bh//2) * scale_y)
                        
                        print(f"\r[识别成功] 读数: {val:03d} | 点击目标: ({click_x}, {click_y})   ", end="")
                        
                        # 警报
                        if val > 120:
                            if not last_alert:
                                print(f"\n!!! 警报 {val} > 120 -> 执行点击 !!!")
                                pyautogui.moveTo(click_x, click_y)
                                pyautogui.click()
                                last_alert = True
                        else:
                            last_alert = False

                        # 调试显示
                        if DEBUG_VIEW:
                            cv2.rectangle(img_bgr, (bx, by), (bx+bw, by+bh), (255, 0, 0), 3) # 蓝框标按钮
                            cv2.rectangle(img_bgr, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 3) # 绿框标LCD

                            preview = cv2.resize(roi_lcd, (200, 100))
                            img_bgr[0:100, 0:200] = preview
                            cv2.putText(img_bgr, f"AI See: {val}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            
                            # 缩小全图显示
                            small = cv2.resize(img_bgr, (0,0), fx=0.5, fy=0.5)
                            cv2.imshow("Bot View", small)
                            if cv2.waitKey(1) == ord('q'): break
                    else:
                         print(f"\r[搜索中] 找到按钮，但没找到LCD", end="")
                         if DEBUG_VIEW:
                             # 标出搜索区域
                             cv2.rectangle(img_bgr, (search_x1, search_y1), (search_x2, search_y2), (0, 0, 255), 2)
                             small = cv2.resize(img_bgr, (0,0), fx=0.5, fy=0.5)
                             cv2.imshow("Bot View", small)
                             if cv2.waitKey(1) == ord('q'): break
                else:
                    print(f"\r[异常] 按钮位置太靠上，无法搜索LCD...", end="")
            else:
                print(f"\r[搜索中] 寻找蓝色按钮...", end="")
                if DEBUG_VIEW:
                    small = cv2.resize(img_bgr, (0,0), fx=0.5, fy=0.5)
                    cv2.imshow("Bot View", small)
                    if cv2.waitKey(1) == ord('q'): break
            
            time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n停止。")
            break

    if DEBUG_VIEW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()