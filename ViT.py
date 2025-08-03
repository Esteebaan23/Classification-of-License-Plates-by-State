import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import pandas as pd
import timm

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

import os
import zipfile
import gdown
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


base_dir = os.getcwd()  
data_dir = os.path.join(base_dir, "dataset")
os.makedirs(data_dir, exist_ok=True)

file_ids = {
    "train": "1dq83d-9twodOj2Z82PpawfD4IAPursgb",
    "val": "14pR8GOc1AvruPeizrjXF-nGX8PcaX82v",
    "test": "1F4HHbp2BzhK55JO0n5N8gC_i1EX088XN"
}


for split, file_id in file_ids.items():
    zip_filename = f"{split}.zip"
    zip_path = os.path.join(data_dir, zip_filename)
    
    print(f"🔽 Downloading {zip_filename}...")
    gdown.download(id=file_id, output=zip_path, quiet=False)

    print(f"🗜️ UnZip {zip_filename}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_path)

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

print("✅ Done...")
print(f"Train dir: {train_dir}")
print(f"Val dir:   {val_dir}")
print(f"Test dir:  {test_dir}")

print("Training...")


def show_random_images(folder, n=5):
    classes = os.listdir(folder)
    for cls in random.sample(classes, n):
        img_paths = glob(os.path.join(folder, cls, "*.jpg"))
        img = Image.open(random.choice(img_paths))
        plt.imshow(img)
        plt.title(f"Class: {cls}")
        plt.axis('off')
        plt.show()

#show_random_images(train_dir, n=3)
def apply_clahe(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final


sample_image_path = glob(os.path.join(train_dir, "*/*.jpg"))[0]
original = Image.open(sample_image_path)
enhanced = apply_clahe(sample_image_path)

from torch.utils.data import Dataset

class PlatesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = sorted(os.listdir(root_dir))
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

        for label in self.labels:
            img_paths = glob(os.path.join(root_dir, label, "*.jpg"))
            self.samples.extend([(path, self.label2idx[label]) for path in img_paths])

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = apply_clahe(img_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
    
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

train_dataset = PlatesDataset(train_dir, transform=transform)
val_dataset = PlatesDataset(val_dir, transform=transform)
test_dataset = PlatesDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def get_vit_model(num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model.to(device)

vit_model = get_vit_model(num_classes=56)

def train_model(model, train_loader, val_loader, device, epochs=7, checkpoint_dir="checkpoints", excel_path="metrics.xlsx"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'loss': total_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    # Save metrics to Excel
    df = pd.DataFrame(history)
    df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")

    return model

vit_model = train_model(vit_model,train_loader,val_loader,device)
torch.save(vit_model.state_dict(), "vit_model_plates.pth")

def evaluate_full_test(model, dataloader, class_names, name="Model", export_dir="eval_results"):
    os.makedirs(export_dir, exist_ok=True)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🔍 Accuracy on Test ({name}): {acc*100:.2f}%")

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=3, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print(f"\n📄 Classification Report ({name}):")
    print(report_df)

    # Save classification report to Excel
    report_path = os.path.join(export_dir, f"{name}_classification_report.xlsx")
    report_df.to_excel(report_path)
    print(f"📁 Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Save confusion matrix to Excel
    cm_path = os.path.join(export_dir, f"{name}_confusion_matrix.xlsx")
    df_cm.to_excel(cm_path)
    print(f"📁 Confusion matrix saved to {cm_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    return y_true, y_pred

y_true, y_pred = evaluate_full_test(vit_model, val_loader, val_dataset.labels, name="ViT")

def export_50_predictions(dataset, y_true, y_pred, title="", export_dir="predicted_samples"):
    os.makedirs(export_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), 50)

    for idx in indices:
        img, label = dataset[idx]
        pred = y_pred[idx]

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)

        plt.figure(figsize=(3, 3))
        plt.imshow(img_np)
        plt.title(f"{title}\nTrue: {dataset.labels[label]} - Pred: {dataset.labels[pred]}")
        plt.axis('off')

        filename = f"{idx:04d}_T-{dataset.labels[label]}_P-{dataset.labels[pred]}.png"
        filepath = os.path.join(export_dir, filename)
        plt.savefig(filepath)
        plt.close()

    print(f"✅ 50 predictions exported to folder: {export_dir}")
    
export_50_predictions(val_dataset, y_true, y_pred, title="ViT")
