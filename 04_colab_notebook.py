# =====================================================================
# CELL 1: SETUP & IMPORTS
# =====================================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully")

# Install required libraries
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q scikit-learn pandas numpy matplotlib seaborn tqdm pillow

print("✓ All packages installed")

# =====================================================================
# CELL 2: IMPORTS & CONFIGURATION
# =====================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from PIL import Image

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {DEVICE}")
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Configuration
CONFIG = {
    'data_path': '/content/drive/MyDrive/dataset_daun_padi',
    'output_path': '/content/results',
    'model_save_path': '/content/results/best_model.pth',
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'image_size': 256,
    'num_classes': 5,
    'class_names': ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut', 'Hispa', 'Healthy'],
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'seed': 42,
    'num_workers': 0,  # Set to 0 untuk Colab
}

# Create output directory
os.makedirs(CONFIG['output_path'], exist_ok=True)
print(f"✓ Output directory: {CONFIG['output_path']}")

# =====================================================================
# CELL 3: HELPER FUNCTIONS
# =====================================================================

def set_seed(seed):
    """Set random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

import copy
import random
set_seed(CONFIG['seed'])

# =====================================================================
# CELL 4: DATASET SPLITTING
# =====================================================================

print("\n📊 === DATASET SPLITTING ===\n")

data_path = Path(CONFIG['data_path'])
if not data_path.exists():
    raise FileNotFoundError(f"Dataset path tidak ditemukan: {data_path}")

class_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
print(f"✓ Ditemukan {len(class_folders)} kelas\n")

all_files = []

for class_folder in class_folders:
    class_name = class_folder.name
    image_files = list(class_folder.glob('*.jpg')) + \
                 list(class_folder.glob('*.JPG')) + \
                 list(class_folder.glob('*.jpeg')) + \
                 list(class_folder.glob('*.JPEG')) + \
                 list(class_folder.glob('*.png')) + \
                 list(class_folder.glob('*.PNG'))
    
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * CONFIG['train_ratio'])
    n_val = int(n_total * CONFIG['val_ratio'])
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"📁 {class_name}")
    print(f"   Total: {n_total:3d} | Train: {len(train_files):3d} | Val: {len(val_files):2d} | Test: {len(test_files):2d}")
    
    for split_type, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            all_files.append({
                'image_path': str(file),
                'label': class_name,
                'split': split_type
            })

df_files = pd.DataFrame(all_files)
print(f"\n✓ Total images: {len(df_files)}")
print(f"  Train: {len(df_files[df_files['split']=='train'])} | Val: {len(df_files[df_files['split']=='val'])} | Test: {len(df_files[df_files['split']=='test'])}\n")

# =====================================================================
# CELL 5: CUSTOM DATASET & TRANSFORMS
# =====================================================================

class RiceDiseaseDataset(Dataset):
    """Custom Dataset untuk Rice Disease Classification"""
    
    def __init__(self, image_paths, labels, transform=None, class_names=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or [str(i) for i in range(len(set(labels)))]
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.labels_idx = [self.label_to_idx[label] for label in labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels_idx[idx]
        return image, label, self.image_paths[idx]

# Transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✓ Transforms defined")

# =====================================================================
# CELL 6: CREATE DATASETS & DATALOADERS
# =====================================================================

# Split data
train_df = df_files[df_files['split'] == 'train']
val_df = df_files[df_files['split'] == 'val']
test_df = df_files[df_files['split'] == 'test']

# Create datasets
train_dataset = RiceDiseaseDataset(
    train_df['image_path'].tolist(),
    train_df['label'].tolist(),
    transform=train_transform,
    class_names=CONFIG['class_names']
)

val_dataset = RiceDiseaseDataset(
    val_df['image_path'].tolist(),
    val_df['label'].tolist(),
    transform=val_test_transform,
    class_names=CONFIG['class_names']
)

test_dataset = RiceDiseaseDataset(
    test_df['image_path'].tolist(),
    test_df['label'].tolist(),
    transform=val_test_transform,
    class_names=CONFIG['class_names']
)

# Create dataloaders
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
    num_workers=CONFIG['num_workers'], pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
    num_workers=CONFIG['num_workers'], pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
    num_workers=CONFIG['num_workers'], pin_memory=pin_memory
)

print("✓ Datasets and DataLoaders created")
print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

# =====================================================================
# CELL 7: BUILD MODEL
# =====================================================================

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.mobilenet = models.mobilenet_v2(weights=weights)
        
        # Freeze early layers
        for param in self.mobilenet.features[:15].parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)

model = MobileNetV2Classifier(num_classes=CONFIG['num_classes'], pretrained=True)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model created (MobileNetV2)")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# =====================================================================
# CELL 8: TRAINING SETUP
# =====================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
}

best_val_acc = 0
best_model_state = None

print("✓ Training setup complete")

# =====================================================================
# CELL 9: TRAINING LOOP
# =====================================================================

print("\n" + "="*60)
print("🚀 === TRAINING MODEL ===")
print("="*60 + "\n")

for epoch in range(CONFIG['epochs']):
    # Training
    model.train()
    train_loss, train_acc = 0, 0
    total_train = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
    for batch_idx, (images, labels, _) in enumerate(train_pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        train_acc += (predicted == labels).sum().item()
        
        train_pbar.set_postfix({
            'loss': f"{train_loss/(batch_idx+1):.4f}",
            'acc': f"{100*train_acc/total_train:.2f}%"
        })
    
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = 100 * train_acc / total_train
    
    # Validation
    model.eval()
    val_loss, val_acc = 0, 0
    total_val = 0
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(val_pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            val_acc += (predicted == labels).sum().item()
            
            val_pbar.set_postfix({
                'loss': f"{val_loss/(batch_idx+1):.4f}",
                'acc': f"{100*val_acc/total_val:.2f}%"
            })
    
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * val_acc / total_val
    
    # Store history
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    # Save best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"✓ Best model saved! Val Acc: {epoch_val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(epoch_val_acc)
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
    print("-" * 60 + "\n")

print("✓ Training completed!")

# =====================================================================
# CELL 10: LOAD BEST MODEL & EVALUATE
# =====================================================================

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"✓ Best model loaded (Val Acc: {best_val_acc:.2f}%)\n")

# Test
print("="*60)
print("📈 === MODEL EVALUATION ===")
print("="*60 + "\n")

model.eval()
all_preds, all_labels = [], []

test_pbar = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for images, labels, _ in test_pbar:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CONFIG['class_names']))

# =====================================================================
# CELL 11: VISUALIZATIONS
# =====================================================================

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training & Validation Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_path'], 'training_history.png'), dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training history plot saved")

# =====================================================================
# CELL 12: CONFUSION MATRIX
# =====================================================================

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CONFIG['class_names'],
            yticklabels=CONFIG['class_names'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_path'], 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

print("✓ Confusion matrix plot saved")

# =====================================================================
# CELL 13: SAVE MODEL & CONFIG
# =====================================================================

# Save model
torch.save(model.state_dict(), CONFIG['model_save_path'])
print(f"✓ Model saved: {CONFIG['model_save_path']}")

# Save config
import json
config_path = os.path.join(CONFIG['output_path'], 'config.json')
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=4)
print(f"✓ Config saved: {config_path}")

# Save metrics
metrics_df = pd.DataFrame({
    'Epoch': range(1, len(history['train_loss']) + 1),
    'Train Loss': history['train_loss'],
    'Train Acc': history['train_acc'],
    'Val Loss': history['val_loss'],
    'Val Acc': history['val_acc'],
})
metrics_df.to_csv(os.path.join(CONFIG['output_path'], 'training_metrics.csv'), index=False)
print(f"✓ Metrics saved")

# =====================================================================
# CELL 14: PIPELINE COMPLETE
# =====================================================================

print("\n" + "="*60)
print("✅ PIPELINE LENGKAP SELESAI!")
print("="*60)
print(f"\nRESULTS SUMMARY:")
print(f"  • Test Accuracy: {test_acc*100:.2f}%")
print(f"  • Best Val Acc: {best_val_acc:.2f}%")
print(f"  • Model: {CONFIG['model_save_path']}")
print(f"  • Results: {CONFIG['output_path']}")
print("\n✓ Semua file sudah tersimpan di /content/results/")
print("✓ Anda bisa download semua hasil training dari sana")

# =====================================================================
# CELL 15: PREDICTION PADA GAMBAR BARU (OPTIONAL)
# =====================================================================

# Uncomment untuk test prediction:
"""
# Prediction function

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_test_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
    
    confidence = probabilities[predicted].item()
    
    print(f"\\nPrediction: {CONFIG['class_names'][predicted.item()]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\\nTop 5 Predictions:")
    for i, prob in enumerate(torch.argsort(probabilities, descending=True)[:5]):
        print(f"  {i+1}. {CONFIG['class_names'][prob.item()]}: {probabilities[prob].item()*100:.2f}%")
    
    # Display
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(f"{CONFIG['class_names'][predicted.item()]} (Confidence: {confidence*100:.1f}%)")
    ax.axis('off')
    plt.show()

# Test pada gambar dari test set
test_image_path = test_dataset.image_paths[0]
predict_image(test_image_path)
"""

print("\n✓ Notebook complete! Selamat atas model yang berhasil dibuat! 🎉")
