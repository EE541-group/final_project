import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Pad
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
import copy
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class ResizePadToSquare:
    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image):
        w, h = img.size
        if h < w:
            new_h = self.target_size
            new_w = int(w * self.target_size / h)
        else:
            new_w = self.target_size
            new_h = int(h * self.target_size / w)
        img = F.resize(img, (new_h, new_w))

        w, h = img.size
        pad_w = self.target_size - w
        pad_h = self.target_size - h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

        return img


DATA_DIR = "./dataset/garbage_classification"
classes = os.listdir(DATA_DIR)
NUM_CLASSES = len(classes)
class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}

random_seed = 42
NUM_SAMPLES_PER_CLASS = 500
TRAIN_RATIO = 0.7
BATCH_SIZE = 32
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OUTPUT_ROOT = './checkpoint/mobileNet'
os.makedirs(OUTPUT_ROOT, exist_ok=True)



train_transform = transforms.Compose([
    ResizePadToSquare(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    ResizePadToSquare(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

all_image_paths = []
all_labels = []

for cls in sorted(classes):
    cls_path = os.path.join(DATA_DIR, cls)
    all_images = os.listdir(cls_path)
    selected_images = random.Random(random_seed).sample(all_images, NUM_SAMPLES_PER_CLASS)
    for img in selected_images:
        all_image_paths.append(os.path.join(cls_path, img))
        all_labels.append(class_to_idx[cls])

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, stratify=all_labels, test_size=1 - TRAIN_RATIO, random_state=random_seed)




LRS = [1e-2, 1e-3, 1e-4]

OPTIMIZERS = {
    'sgd':  lambda params, lr: optim.SGD(params,  lr=lr, momentum=0.9, weight_decay=1e-4),
    'adam': lambda params, lr: optim.Adam(params, lr=lr, weight_decay=1e-4),
}

STRATEGIES = {
    'fc_only_then_full': 'freeze_backbone_then_full', 
    'fc_only':           'freeze_backbone',         
}

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform=None, loader=default_loader):
        self.paths = paths; self.labels = labels
        self.transform = transform; self.loader = loader
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = self.loader(self.paths[idx])
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

train_dataset = ImageListDataset(train_paths, train_labels, transform=train_transform)
val_dataset   = ImageListDataset(val_paths,   val_labels,   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=10)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=10)


def build_mobilenet(num_classes, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, preds, targets = 0, [], []
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds += torch.argmax(logits,1).cpu().tolist()
        targets += y.cpu().tolist()
    return running_loss/len(loader.dataset), accuracy_score(targets,preds)

def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss, preds, targets = 0, [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds += torch.argmax(logits,1).cpu().tolist()
            targets += y.cpu().tolist()
    return running_loss/len(loader.dataset), accuracy_score(targets,preds), preds, targets


results = {}

for lr in LRS:
    for opt_name, opt_fn in OPTIMIZERS.items():
        for strat_key, strat in STRATEGIES.items():
            exp_name = f"mbv2_lr{lr}_{opt_name}_{strat_key}"
            print(f"\n=== Experiment: {exp_name} ===")
            model = build_mobilenet(NUM_CLASSES, pretrained=True).to(DEVICE)

            if strat == 'freeze_backbone':
                for name,param in model.named_parameters():
                    param.requires_grad = ('classifier' in name)
            else:  
                for name,param in model.named_parameters():
                    param.requires_grad = ('classifier' in name)

            optimizer = opt_fn(filter(lambda p:p.requires_grad, model.parameters()), lr)
            criterion = nn.CrossEntropyLoss()

            history = {'train_loss':[], 'val_loss':[],
                       'train_acc':[],  'val_acc':[]}

            for epoch in range(NUM_EPOCHS):
                if strat=='freeze_backbone_then_full' and epoch==NUM_EPOCHS//2:
                    for p in model.parameters(): p.requires_grad = True
                    optimizer = opt_fn(model.parameters(), lr*0.1)

                t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
                v_loss, v_acc, v_preds, v_trues = eval_one_epoch(model, val_loader, criterion)

                history['train_loss'].append(t_loss)
                history['val_loss'].append(v_loss)
                history['train_acc'].append(t_acc)
                history['val_acc'].append(v_acc)

                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"TLoss={t_loss:.4f}, TAcc={t_acc:.4f} | "
                      f"VLoss={v_loss:.4f}, VAcc={v_acc:.4f}")

            ckpt_path = os.path.join(OUTPUT_ROOT, f"{exp_name}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
                'exp_name': exp_name
            }, ckpt_path)

            cm = confusion_matrix(v_trues, v_preds)
            # 绘制
            plt.figure(figsize=(6,6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"{exp_name} Confusion Matrix")
            plt.colorbar()
            ticks = np.arange(NUM_CLASSES)
            plt.xticks(ticks, ticks, rotation=45)
            plt.yticks(ticks, ticks)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            cm_path = os.path.join(OUTPUT_ROOT, f"{exp_name}_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

            results[exp_name] = history


for exp_name, hist in results.items():
    epochs = range(1, NUM_EPOCHS+1)
    plt.figure(figsize=(12,4))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['train_loss'], label='Train Loss')
    plt.plot(epochs, hist['val_loss'],   label='Val   Loss')
    plt.title(f"{exp_name} Loss")
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, hist['train_acc'], label='Train Acc')
    plt.plot(epochs, hist['val_acc'],   label='Val   Acc')
    plt.title(f"{exp_name} Accuracy")
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
