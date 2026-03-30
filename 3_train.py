import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

#模型定义
class ScadaLCDNet(nn.Module):
    def __init__(self, num_classes=151):
        super().__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器部分
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
        return self.classifier(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((120, 240)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((120, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    if not os.path.exists('dataset_lcd'):
        print("Error: 'dataset_lcd' folder not found. Please run 2_data_gen.py first.")
        return

    full_dataset = datasets.ImageFolder('dataset_lcd', transform=train_transform)
    
    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 修正验证集的 transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    model = ScadaLCDNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0

    epochs = 20
    
    for epoch in range(epochs):
        #训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * train_correct / train_total:.2f}%"})
        
        #验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("model_out", exist_ok=True)
            torch.save(model.state_dict(), "model_out/scada_lcd_net_best.pth")
            print(f"--> New Best Model Saved! ({val_acc:.2f}%)")

    #保存最终模型
    torch.save(model.state_dict(), "model_out/scada_lcd_net.pth")
    print("Training completed. Best Val Acc: {:.2f}%".format(best_val_acc))

if __name__ == "__main__":
    train()