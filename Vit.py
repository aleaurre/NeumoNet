import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16
import torchvision.transforms as T

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score




class CFG:
    seed = 42
    # tamaño de imagen que espera ViT preentrenado de torchvision
    img_size = 224
    # cuántas imágenes se procesan juntas
    batch_size = 16
    # cuántas veces recorrer dataset completo
    epochs = 15
    # learning rate → velocidad de aprendizaje
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
# (hace que dos ejecuciones den resultados parecidos)
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)


# ============================================================
# ARMAR DATASET
# Busca imágenes ya procesadas por standardization.py
# ============================================================

def build_dataframe():
    rows = []
    normal_dir = Path("./assets/post_norm")
    pneumonia_dir = Path("./assets/post_pneu")

    # -----------------------------------
    # NORMAL = label 0
    # -----------------------------------
    for img_path in normal_dir.glob("*"):
        rows.append({
            "path": str(img_path),
            "label": 0
        })

    # -----------------------------------
    # PNEUMONIA = label 1
    # -----------------------------------
    for img_path in pneumonia_dir.glob("*"):
        rows.append({
            "path": str(img_path),
            "label": 1
        })

    df = pd.DataFrame(rows)

    return df


# ============================================================
# TRANSFORMACIONES
# Preparan imágenes antes de entrar al modelo
# ============================================================

def build_transforms():

    train_transforms = T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),

        # 50% chance de invertir horizontalmente
        # ayuda a generalizar
        T.RandomHorizontalFlip(0.5),

        # convertir imagen a tensor pytorch
        T.ToTensor(),

        # radiografía es grayscale (1 canal)
        # ViT espera RGB (3 canales)
        # esto duplica:
        # [gray]
        # en:
        # [gray gray gray]
        T.Lambda(lambda x: x.repeat(3, 1, 1)),

        # normalizar valores entre distribución estándar
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


    # validation no necesita augmentation
    val_transforms = T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return train_transforms, val_transforms


# ============================================================
# DATASET CLASS
# PyTorch necesita esta estructura
# ============================================================

class XRayDataset(Dataset):

    # constructor
    def __init__(self, dataframe, transforms):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms


    # cantidad de imágenes
    def __len__(self):
        return len(self.df)


    # obtiene UNA imagen
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # abrir imagen
        # convert("L") = grayscale
        image = Image.open(row["path"]).convert("L")
        # aplicar transformaciones
        image = self.transforms(image)

        # convertir label a tensor
        # 0 → normal
        # 1 → pneumonia
        label = torch.tensor(row["label"]).float()

        return image, label


# ============================================================
# CREAR MODELO
# ============================================================

def build_model():

    # cargar Vision Transformer preentrenado
    # pretrained en ImageNet
    model = vit_b_16(weights="IMAGENET1K_V1")

    # obtener cantidad de features de salida
    in_features = model.heads.head.in_features


    # reemplazar cabeza final
    # original:
    # clasifica miles de objetos
    # nuevo:
    # clasifica 2 clases
    # usamos 1 salida porque BCEWithLogits
    model.heads.head = nn.Linear(in_features, 1)

    return model


# ============================================================
# ENTRENAR UNA EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, criterion):

    # modo entrenamiento
    model.train()
    running_loss = 0

    # recorrer batches
    for images, labels in loader:
        # mover a GPU/CPU
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)
      
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # promedio del error
    return running_loss / len(loader)


# ============================================================
# VALIDATION
# evalúa modelo sin entrenar
# ============================================================

def validate(model, loader):

    # modo evaluación
    model.eval()

    all_probs = []
    all_targets = []

    # no calcular gradientes
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(CFG.device)

            # forward pass
            outputs = model(images).squeeze(1)

            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # threshold
    preds = (all_probs > 0.5).astype(int)
    f1 = f1_score(all_targets, preds)
    return f1


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def run_training():

    df = build_dataframe()

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=CFG.seed,
        stratify=df["label"]
    )

    # obtener transforms
    train_tfms, val_tfms = build_transforms()

    # crear datasets
    train_ds = XRayDataset(
        train_df,
        train_tfms
    )

    val_ds = XRayDataset(
        val_df,
        val_tfms
    )

    # dataloaders
    # batch loading
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False
    )

    model = build_model().to(CFG.device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr
    )

    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------
    # entrenamiento principal
    # ------------------------------------------------
    for epoch in range(CFG.epochs):

        # entrenar
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        # validar
        val_f1 = validate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch+1}/{CFG.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"F1: {val_f1:.4f}"
        )


    # guardar pesos entrenados
    torch.save(
        model.state_dict(),
        "vit_pneumonia.pth"
    )

    print("Training complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # empezar entrenamiento
    run_training()