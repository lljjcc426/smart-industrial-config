# 工业组态智能增强系统

一个面向《神经网络与深度学习》课程实验的项目：通过模拟工业 SCADA 压力监控界面，自动生成 LCD 数字图像数据集，训练卷积神经网络识别读数，并在读数超过阈值时自动触发泄压按钮，实现“视觉识别 + 智能决策 + 自动控制”的闭环演示。

## 项目简介

本项目围绕一个简化的工业压力监控场景展开：

- `1_simulator.py` 使用 `pygame` 构建工业监控界面，实时模拟压力上升与报警状态。
- `2_data_gen.py` 自动生成 LCD 读数图像数据集，并加入位移、噪声、模糊、亮度扰动等增强。
- `3_train.py` 使用 `PyTorch` 训练 `ScadaLCDNet` 模型，对 `000` 到 `150` 的数字读数进行分类。
- `4_main_bot.py` 通过 `mss + OpenCV` 截屏和定位 LCD 区域，调用训练好的模型识别读数，并在压力超过阈值时用 `pyautogui` 自动点击泄压按钮。

项目适合用于展示以下能力：

- 深度学习项目的完整流程：数据构建、模型训练、推理部署
- 计算机视觉与自动化控制的联动
- 封闭界面上的“非侵入式”智能增强思路

## 运行效果

系统的核心逻辑是：

1. 仿真器持续显示压力数值。
2. Bot 实时截取屏幕并定位 LCD 区域。
3. 模型识别当前压力值。
4. 当读数大于 `120` 时，自动点击 `EMERGENCY VENT` 按钮执行泄压。

当前仓库根目录下已经提供训练好的模型：

- `model_out/scada_lcd_net_best.pth`
- `model_out/scada_lcd_net.pth`

因此如果只是演示效果，可以不重新训练，直接启动仿真器和 Bot。

## 项目结构

```text
.
├─ 1_simulator.py                  # 工业压力监控界面模拟
├─ 2_data_gen.py                   # LCD 数字图像数据集生成
├─ 3_train.py                      # ScadaLCDNet 模型训练
├─ 4_main_bot.py                   # 屏幕识别与自动点击控制
├─ model_out/                      # 训练得到的模型权重
├─ 神经网络实验报告-李珈辰-2023213809.doc
├─ 神经网络实验报告-李珈辰-2023213809.pdf
├─ deep/                           # 项目备份/整理目录
└─ deep.zip                        # 项目压缩包
```

运行 `2_data_gen.py` 后会在当前目录生成：

```text
dataset_lcd/
├─ 000/
├─ 001/
├─ ...
└─ 150/
```

## 环境依赖

建议使用 Python 3.10 及以上版本，操作系统以 Windows 为主。

安装依赖：

```bash
pip install torch torchvision opencv-python pillow mss pygame pyautogui numpy tqdm
```

## 快速开始

### 1. 启动仿真界面

```bash
python 1_simulator.py
```

界面会持续模拟压力变化，并显示一个蓝色的 `EMERGENCY VENT` 按钮。

### 2. 直接运行智能 Bot

确保仿真界面已经打开，再执行：

```bash
python 4_main_bot.py
```

程序会：

- 自动搜索蓝色按钮位置
- 在按钮上方搜索黑色 LCD 屏区域
- 识别 LCD 中的三位数字
- 当读数大于 `120` 时自动点击按钮

按 `q` 可关闭 Bot 的调试窗口。

## 从零复现实验

如果你希望完整复现“数据生成 -> 模型训练 -> 部署控制”的流程，可以按下面顺序执行。

### 1. 生成数据集

```bash
python 2_data_gen.py
```

默认会为 `000-150` 每个类别生成 `500` 张图片，共 `151` 类。

### 2. 训练模型

```bash
python 3_train.py
```

训练脚本会：

- 自动划分训练集和验证集
- 训练 `ScadaLCDNet`
- 将最佳模型保存为 `model_out/scada_lcd_net_best.pth`
- 将最终模型保存为 `model_out/scada_lcd_net.pth`

### 3. 联动演示

```bash
python 1_simulator.py
python 4_main_bot.py
```

建议分两个终端分别启动。

## 模型说明

`ScadaLCDNet` 是一个针对 LCD 数字识别任务设计的轻量 CNN，主要包含：

- 4 层卷积特征提取
- Batch Normalization
- ReLU + MaxPooling
- Global Average Pooling
- 两层全连接分类器

分类目标为 `151` 类，对应读数 `000-150`。

## 注意事项

- `4_main_bot.py` 依赖屏幕截图和鼠标控制，建议在 Windows 环境运行。
- 运行 Bot 时请保证仿真窗口完整可见，避免被其他窗口遮挡。
- 如果没有找到模型文件，请先运行 `3_train.py`，或确认 `model_out/scada_lcd_net_best.pth` 是否存在。
- 如果没有数据集目录，请先运行 `2_data_gen.py`。
- `pyautogui` 会真实移动并点击鼠标，演示时请避免误操作其他窗口。

## 项目定位

这个仓库更偏向课程实验与原型验证，重点不在工业现场部署，而在于展示如何把：

- 仿真环境
- 合成数据集
- 卷积神经网络识别
- 屏幕级自动化控制

组合成一个可运行、可演示、可复现的智能增强系统。
