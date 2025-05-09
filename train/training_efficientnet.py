import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
import torchvision.transforms.functional as F
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


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
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img = F.pad(img, (left, top, right, bottom), fill=self.fill)
        return img


def build_resnet(num_classes, backbone="resnet50", pretrained=True):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif backbone == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    else:
        model = models.resnet50(pretrained=pretrained)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def build_efficientnet(num_classes, backbone="b0", pretrained=True):
    # torchvision supports efficientnet_b0…_b7
    fn = getattr(models, f"efficientnet_{backbone}")
    model = fn(pretrained=pretrained)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model

DATA_DIR       = "./dataset/garbage_classification"
OUTPUT_ROOT    = "./checkpoint/efficientnet"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SEED           = 42
NUM_PER_CLASS  = 500
TRAIN_RATIO    = 0.7
BATCH_SIZE     = 32
NUM_EPOCHS     = 20
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LRS            = [1e-2, 1e-3, 1e-4]
OPTIMIZERS     = {
    'sgd':  lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4),
    'adam': lambda params, lr: optim.Adam(params, lr=lr, weight_decay=1e-4),
}
STRATEGIES     = {
    'freeze_fc': 'freeze_backbone',
    'finetune':  'freeze_backbone_then_full',
}

BACKBONES     = [
    ("efficientnet", "b0")
]


random.seed(SEED)
classes = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(classes)
cls_to_idx = {c:i for i,c in enumerate(classes)}

all_paths, all_labels = [], []
for cls in classes:
    folder = os.path.join(DATA_DIR, cls)
    imgs = os.listdir(folder)
    selected = random.sample(imgs, NUM_PER_CLASS)
    for img in selected:
        all_paths.append(os.path.join(folder, img))
        all_labels.append(cls_to_idx[cls])

train_paths, val_paths, train_lbls, val_lbls = train_test_split(
    all_paths, all_labels,
    stratify=all_labels,
    test_size=1-TRAIN_RATIO,
    random_state=SEED
)

transform_train = transforms.Compose([
    ResizePadToSquare(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
transform_val = transforms.Compose([
    ResizePadToSquare(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

train_ds = ImageListDataset(train_paths, train_lbls, transform_train)
val_ds   = ImageListDataset(val_paths,   val_lbls,   transform_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds, trues = 0, [], []
    for x,y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds += torch.argmax(logits,1).cpu().tolist()
        trues += y.cpu().tolist()
    return total_loss/len(loader.dataset), accuracy_score(trues,preds)

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds, trues = 0, [], []
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds += torch.argmax(logits,1).cpu().tolist()
            trues += y.cpu().tolist()
    return total_loss/len(loader.dataset), accuracy_score(trues,preds), preds, trues


results = {}

for kind,name in BACKBONES:
    for lr in LRS:
        for opt_name,opt_fn in OPTIMIZERS.items():
            for strat_key, strat in STRATEGIES.items():
                exp = f"{kind}_{name}_lr{lr}_{opt_name}_{strat_key}"
                print(f"\n=== Experiment: {exp} ===")

                if kind=="resnet":
                    model = build_resnet(NUM_CLASSES, backbone=name).to(DEVICE)
                else:
                    model = build_efficientnet(NUM_CLASSES, backbone=name).to(DEVICE)

                freeze_terms = ["fc","classifier"]
                if strat=="freeze_backbone":
                    for n,p in model.named_parameters():
                        p.requires_grad = any(term in n for term in freeze_terms)
                else: 
                    for n,p in model.named_parameters():
                        p.requires_grad = any(term in n for term in freeze_terms)

                optimizer = opt_fn(filter(lambda p:p.requires_grad, model.parameters()), lr)
                criterion = nn.CrossEntropyLoss()
                history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}

                for epoch in range(NUM_EPOCHS):
                    if strat=="freeze_backbone_then_full" and epoch==NUM_EPOCHS//2:
                        for p in model.parameters():
                            p.requires_grad = True
                        optimizer = opt_fn(model.parameters(), lr*0.1)

                    t_loss,t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
                    v_loss,v_acc,v_preds,v_trues = eval_one_epoch(model, val_loader, criterion)

                    history['train_loss'].append(t_loss)
                    history['val_loss'].append(v_loss)
                    history['train_acc'].append(t_acc)
                    history['val_acc'].append(v_acc)

                    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                          f"TLoss={t_loss:.4f}, TAcc={t_acc:.4f} | "
                          f"VLoss={v_loss:.4f}, VAcc={v_acc:.4f}")

                torch.save({
                    'model_state':    model.state_dict(),
                    'optimizer_state':optimizer.state_dict(),
                    'history':        history,
                    'exp':            exp
                }, os.path.join(OUTPUT_ROOT, f"{exp}.pth"))

                cm = confusion_matrix(v_trues, v_preds)
                plt.figure(figsize=(6,6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f"{exp} Confusion Matrix")
                plt.colorbar()
                ticks = np.arange(NUM_CLASSES)
                plt.xticks(ticks, ticks, rotation=45)
                plt.yticks(ticks, ticks)
                thresh = cm.max()/2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j,i,cm[i,j],ha="center",va="center",
                                 color="white" if cm[i,j]>thresh else "black")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_ROOT, f"{exp}_cm.png"))
                plt.close()

                results[exp] = history

for exp, hist in results.items():
    epochs = range(1, NUM_EPOCHS+1)
    plt.figure(figsize=(12,4))
    # loss
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['train_loss'], label='Train Loss')
    plt.plot(epochs, hist['val_loss'],   label='Val Loss')
    plt.title(f"{exp} Loss"); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()
    # acc
    plt.subplot(1,2,2)
    plt.plot(epochs, hist['train_acc'], label='Train Acc')
    plt.plot(epochs, hist['val_acc'],   label='Val Acc')
    plt.title(f"{exp} Acc"); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
