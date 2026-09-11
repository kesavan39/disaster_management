"""
================================================================================
ROLE 03: DEEP LEARNING ENGINEER
Disaster Response Coordination System - Stage 02 Deep Learning
================================================================================
Responsibilities:
- Build & train PyTorch CNN for drone image flood classification (Clear vs Flooded).
- Build & train PyTorch Multi-Horizon LSTM for 2 to 10 hour river water level forecasting.
- Serialize trained neural network weights & metadata into DL/saved_models/.
================================================================================
"""

import os
import json
import joblib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define PyTorch Deep Residual CNN Architecture for Drone Image Classification (FloodResNet)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class FloodResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(FloodResNet, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

# Alias for backward compatibility
DroneImageCNN = FloodResNet

# Define PyTorch Multi-Horizon LSTM Architecture for 2-10 hr Water Level Forecasting
class WaterLevelLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, forecast_horizon=10):
        super(WaterLevelLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        final_state = lstm_out[:, -1, :]
        out = self.fc(final_state)
        return out

# PyTorch Image Dataset Wrapper with Data Augmentation & ImageNet Normalization
class AerialImageDataset(Dataset):
    def __init__(self, flooded_dir, clear_dir, target_size=(128, 128), augment=True):
        self.image_paths = []
        self.labels = []
        self.target_size = target_size
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if os.path.exists(clear_dir):
            for f in sorted(os.listdir(clear_dir)):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(clear_dir, f))
                    self.labels.append(0) # 0 = Clear

        if os.path.exists(flooded_dir):
            for f in sorted(os.listdir(flooded_dir)):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(flooded_dir, f))
                    self.labels.append(1) # 1 = Flooded

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB').resize(self.target_size)
        
        # PIL Data Augmentation during training
        if self.augment:
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.rand() > 0.5:
                rot_angle = int(np.random.choice([90, 180, 270]))
                img = img.rotate(rot_angle)

        arr = np.array(img, dtype=np.float32) / 255.0 # Normalize 0-1
        arr = (arr - self.mean) / self.std # ImageNet standardization
        arr = np.transpose(arr, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class DLEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "Data")
        self.dl_dir = os.path.join(self.base_dir, "DL")
        self.models_dir = os.path.join(self.dl_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_cnn(self, epochs=20, batch_size=32):
        """Trains FloodResNet classifier on aerial drone images with train/val split & CosineLR."""
        print(f"[DL Engineer] Training PyTorch FloodResNet Classifier on {self.device}...")
        
        flooded_dir = os.path.join(self.data_dir, "images", "flooded")
        clear_dir = os.path.join(self.data_dir, "images", "clear")
        
        full_dataset = AerialImageDataset(flooded_dir, clear_dir, target_size=(128, 128), augment=True)
        if len(full_dataset) == 0:
            raise ValueError("No images found in dataset directories! Run DL/01_data_engineer.py first.")

        # 80/20 Train/Val Split
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = FloodResNet(num_classes=2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        best_model_weights = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(y_batch)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += len(y_batch)

            scheduler.step()

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                    out_v = model(X_v)
                    v_loss = criterion(out_v, y_v)
                    val_loss += v_loss.item() * len(y_v)
                    v_preds = torch.argmax(out_v, dim=1)
                    val_correct += (v_preds == y_v).sum().item()
                    val_total += len(y_v)

            epoch_train_acc = (train_correct / train_total) * 100.0
            epoch_val_acc = (val_correct / val_total) * 100.0
            
            if epoch_val_acc >= best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_weights = model.state_dict()

            if (epoch + 1) % 4 == 0 or epoch == epochs - 1:
                print(f"[DL Engineer] ResNet Epoch [{epoch+1}/{epochs}] -> Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        cnn_path = os.path.join(self.models_dir, "cnn_model.pth")
        torch.save(best_model_weights or model.state_dict(), cnn_path)
        print(f"[DL Engineer] Trained FloodResNet Model saved to {cnn_path} (Best Val Accuracy: {best_val_acc:.2f}%)")
        return model

    def train_lstm(self, epochs=15, batch_size=64):
        """Trains PyTorch Multi-Horizon LSTM on 24-hr sequence windows for 10-hr forecast."""
        print(f"[DL Engineer] Training PyTorch Multi-Horizon LSTM Forecaster on {self.device}...")
        
        seq_path = os.path.join(self.dl_dir, "lstm_sequences.npz")
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"Sequences file {seq_path} not found. Run DL/01_data_engineer.py first.")

        data = np.load(seq_path)
        X_train, y_train = data['X_train'], data['y_train']
        
        # Subsample for efficient training while preserving full distribution
        if len(X_train) > 10000:
            idx = np.random.choice(len(X_train), 10000, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = WaterLevelLSTM(input_size=3, hidden_size=64, num_layers=2, forecast_horizon=10).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total = 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(y_b)
                total += len(y_b)

            epoch_rmse = np.sqrt(total_loss / total)
            if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
                print(f"[DL Engineer] LSTM Epoch [{epoch+1}/{epochs}] -> Training RMSE: {epoch_rmse:.4f}")

        lstm_path = os.path.join(self.models_dir, "lstm_model.pth")
        torch.save(model.state_dict(), lstm_path)
        print(f"[DL Engineer] Trained LSTM Model saved to {lstm_path}")
        return model

    def run_training(self):
        """Runs training for both CNN and LSTM models."""
        print("==========================================================")
        print("[DL Engineer] Starting Deep Learning Model Training...")
        print("==========================================================")
        
        cnn_model = self.train_cnn(epochs=10)
        lstm_model = self.train_lstm(epochs=15)
        
        meta = {
            "cnn_architecture": "DroneImageCNN (3 Conv2D layers + MaxPool + Dropout + FC)",
            "lstm_architecture": "WaterLevelLSTM (2-layer LSTM + FC 10-horizon output)",
            "device_used": str(self.device),
            "status": "TRAINED_AND_SAVED"
        }
        with open(os.path.join(self.models_dir, "dl_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print("[DL Engineer] Deep Learning model training completed successfully!\n")
        return cnn_model, lstm_model

if __name__ == "__main__":
    engineer = DLEngineer()
    engineer.run_training()
