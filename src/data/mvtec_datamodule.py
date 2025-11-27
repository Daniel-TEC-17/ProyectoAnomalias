"""
MVTec AD DataModule para PyTorch Lightning
Carga solo imágenes 'good' para train y validation
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


class MVTecDataset(Dataset):
    """Dataset para MVTec AD con imágenes procesadas."""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            data_dir: directorio base (ej: DATASET_128x128)
            split: 'train', 'validation', o 'test'
            transform: transformaciones de torchvision
            class_names: lista de nombres de clases
        """
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.class_names = class_names or []
        
        # Cargar imágenes
        self.images = []
        self.labels = []
        self.is_anomaly = []  # Para test set
        
        self._load_images()
        
    def _load_images(self):
        """Carga rutas de imágenes y sus etiquetas."""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Obtener todas las imágenes del split
        for img_path in sorted(split_dir.glob("*.*")):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            
            # Extraer información del nombre: dataset_split_class_id.ext
            # Ejemplo: cable_train_good_000.png
            parts = img_path.stem.split('_')
            
            if len(parts) >= 3:
                dataset_name = parts[0]  # cable
                split_name = parts[1]    # train
                class_name = parts[2]    # good
                
                # Asignar label basado en el dataset
                if dataset_name in self.class_names:
                    label = self.class_names.index(dataset_name)
                else:
                    label = 0  # Default
                
                # Determinar si es anomalía (solo relevante para test)
                is_anomaly = (class_name != 'good')
                
                self.images.append(str(img_path))
                self.labels.append(label)
                self.is_anomaly.append(is_anomaly)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: tensor de la imagen
            label: clase del dataset (0-9)

        Note: `is_anomaly` is stored internally but is not returned
        to keep batches compatible with models that expect `(image, label)`.
        If you need the anomaly flag for specific evaluation code, use
        the dataset's `is_anomaly` attribute or access the image path.
        """
        # Cargar imagen
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        # Keep anomaly flag internally (for external evaluation), but
        # return only (image, label) so training/validation/test loops
        # that expect two-item batches work without unpack errors.
        return image, label


class MVTecDataModule(pl.LightningDataModule):
    """DataModule de PyTorch Lightning para MVTec AD."""
    
    def __init__(
        self,
        data_dir: str,
        class_names: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: Tuple[int, int] = (128, 128)
    ):
        """
        Args:
            data_dir: directorio base con train/validation/test
            class_names: lista de nombres de las 10 clases
            batch_size: tamaño del batch
            num_workers: workers para DataLoader
            pin_memory: usar pin_memory para GPU
            image_size: tamaño de las imágenes (ya deberían estar resized)
        """
        super().__init__()
        self.data_dir = data_dir
        self.class_names = class_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Preparar datasets."""
        
        # Transformaciones para train (con augmentation)
        train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Transformaciones para val/test (sin augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Crear datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MVTecDataset(
                self.data_dir, 
                split="train", 
                transform=train_transform,
                class_names=self.class_names
            )
            self.val_dataset = MVTecDataset(
                self.data_dir, 
                split="validation", 
                transform=eval_transform,
                class_names=self.class_names
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = MVTecDataset(
                self.data_dir, 
                split="test", 
                transform=eval_transform,
                class_names=self.class_names
            )
    
    def train_dataloader(self) -> DataLoader:
        """DataLoader para entrenamiento."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Evita batches incompletos
        )
    
    def val_dataloader(self) -> DataLoader:
        """DataLoader para validación."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """DataLoader para test."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_num_samples(self) -> dict:
        """Obtener número de muestras por split."""
        return {
            'train': len(self.train_dataset) if self.train_dataset else 0,
            'val': len(self.val_dataset) if self.val_dataset else 0,
            'test': len(self.test_dataset) if self.test_dataset else 0
        }