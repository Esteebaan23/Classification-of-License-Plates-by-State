import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from glob import glob
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import timm

# Device
device = "cpu"
from torchvision.models import resnet50

resnet_model = resnet50(num_classes=56)

resnet_model.load_state_dict(torch.load('resnet50_plates.pth', map_location=torch.device('cpu')))
resnet_model.eval().to(device)

from torchvision.models import densenet121

densenet_model = densenet121(num_classes=56)
densenet_model.load_state_dict(torch.load('densenet121_model_plates.pth', map_location=torch.device('cpu')))
densenet_model.eval().to(device)

import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        features = [x]
        for layer in self.block:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class CustomDeepCNN(nn.Module):
    def __init__(self, num_classes=56):
        super(CustomDeepCNN, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = DenseBlock(64, growth_rate=32, num_layers=4)
        self.trans1 = TransitionLayer(64 + 4 * 32, 128)

        self.block2 = DenseBlock(128, growth_rate=32, num_layers=6)
        self.trans2 = TransitionLayer(128 + 6 * 32, 256)

        self.block3 = DenseBlock(256, growth_rate=32, num_layers=8)
        self.trans3 = TransitionLayer(256 + 8 * 32, 512)

        self.final_bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.relu(self.final_bn(x))
        x = self.global_pool(x)
        x = self.classifier(x)
        return x



own_model = CustomDeepCNN()
own_model.load_state_dict(torch.load('model_plates2.pth', map_location=torch.device('cpu')))
own_model.eval().to(device)

def apply_clahe(img: Image.Image) -> np.ndarray:
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return img_rgb


with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

print("Clases cargadas:", class_names)

# Transforms for Resnet and Own Model
transform_resnet = transforms.Compose([
    transforms.Resize((128, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

#Transforms for ViT
transform_ViT = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

#Transforms for Own Model
transform_Own = transforms.Compose([
    transforms.Resize((224, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

import streamlit as st
# --------------------------
# Interfaz Streamlit
# --------------------------
st.title("Classification of License Plates by State")

uploaded_file = st.file_uploader("Select an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    # load image
    image = Image.open(uploaded_file).convert("RGB")
    image = apply_clahe(image)
    image = Image.fromarray(image)

    st.image(image, caption="Image", use_container_width=True)

    # transform image
    input_tensor_resnet = transform_resnet(image).unsqueeze(0).to(device)
    #input_tensor_Vit = transform_ViT(image).unsqueeze(0).to(device)
    input_tensor_Own = transform_Own(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred_resnet = resnet_model(input_tensor_resnet)
        pred_denesnet = densenet_model(input_tensor_resnet)
        #pred_vit = vit_model(input_tensor_Vit)
        pred_own = own_model(input_tensor_Own)

    # Get label
    pred_label_resnet = class_names[torch.argmax(pred_resnet).item()]
    pred__label_denesnet = class_names[torch.argmax(pred_denesnet).item()]
   # pred_label_vit = class_names[torch.argmax(pred_vit).item()]
    pred_label_own = class_names[torch.argmax(pred_own).item()]

    # Show results
    st.markdown("### Predictions:")
    st.write(f"- **ResNet50**: {pred_label_resnet}")
    st.write(f"- **DenseNet121**: {pred__label_denesnet}")
    #st.write(f"- **ViT**: {pred_label_vit}")
    st.write(f"- **Own CNN**: {pred_label_own}")


import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from glob import glob
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import timm

# Device
device = "cpu"
from torchvision.models import resnet50

resnet_model = resnet50(num_classes=56)

resnet_model.load_state_dict(torch.load('resnet50_plates.pth', map_location=torch.device('cpu')))
resnet_model.eval().to(device)

from torchvision.models import densenet121

densenet_model = densenet121(num_classes=56)
densenet_model.load_state_dict(torch.load('densenet121_model_plates.pth', map_location=torch.device('cpu')))
densenet_model.eval().to(device)

import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        features = [x]
        for layer in self.block:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class CustomDeepCNN(nn.Module):
    def __init__(self, num_classes=56):
        super(CustomDeepCNN, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = DenseBlock(64, growth_rate=32, num_layers=4)
        self.trans1 = TransitionLayer(64 + 4 * 32, 128)

        self.block2 = DenseBlock(128, growth_rate=32, num_layers=6)
        self.trans2 = TransitionLayer(128 + 6 * 32, 256)

        self.block3 = DenseBlock(256, growth_rate=32, num_layers=8)
        self.trans3 = TransitionLayer(256 + 8 * 32, 512)

        self.final_bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.relu(self.final_bn(x))
        x = self.global_pool(x)
        x = self.classifier(x)
        return x



own_model = CustomDeepCNN()
own_model.load_state_dict(torch.load('model_plates2.pth', map_location=torch.device('cpu')))
own_model.eval().to(device)

def apply_clahe(img: Image.Image) -> np.ndarray:
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return img_rgb


with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

print("Clases cargadas:", class_names)

# Transforms for Resnet and Own Model
transform_resnet = transforms.Compose([
    transforms.Resize((128, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

#Transforms for ViT
transform_ViT = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

#Transforms for Own Model
transform_Own = transforms.Compose([
    transforms.Resize((224, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

import streamlit as st
# --------------------------
# Interfaz Streamlit
# --------------------------
st.title("Classification of License Plates by State")

uploaded_file = st.file_uploader("Select an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    # load image
    image = Image.open(uploaded_file).convert("RGB")
    image = apply_clahe(image)
    image = Image.fromarray(image)

    st.image(image, caption="Image", use_container_width=True)

    # transform image
    input_tensor_resnet = transform_resnet(image).unsqueeze(0).to(device)
    #input_tensor_Vit = transform_ViT(image).unsqueeze(0).to(device)
    input_tensor_Own = transform_Own(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred_resnet = resnet_model(input_tensor_resnet)
        pred_denesnet = densenet_model(input_tensor_resnet)
        #pred_vit = vit_model(input_tensor_Vit)
        pred_own = own_model(input_tensor_Own)

    # Get label
    pred_label_resnet = class_names[torch.argmax(pred_resnet).item()]
    pred__label_denesnet = class_names[torch.argmax(pred_denesnet).item()]
   # pred_label_vit = class_names[torch.argmax(pred_vit).item()]
    pred_label_own = class_names[torch.argmax(pred_own).item()]

    # Show results
    st.markdown("### Predictions:")
    st.write(f"- **ResNet50**: {pred_label_resnet}")
    st.write(f"- **DenseNet121**: {pred__label_denesnet}")
    #st.write(f"- **ViT**: {pred_label_vit}")
    st.write(f"- **Own CNN**: {pred_label_own}")

