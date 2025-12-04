"""
MVTec AD DataModule para PyTorch Lightning
Compatible con la estructura física: train/validation/test
"""
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
            class_names: lista de nombres de clases (para mapear a labels)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.class_names = class_names or []
        self.image_paths = [] 
        self.labels = []
        self.is_anomaly = []
        
        self._load_images()
        
    def _load_images(self):
        """Carga rutas de imágenes y sus etiquetas."""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            # Fallback por si acaso, pero tu script ya creó las carpetas
            print(f"⚠️ Alerta: No existe {split_dir}, buscando en raíz...")
            split_dir = self.data_dir
        
        # Buscar imágenes
        valid_exts = {'.png', '.jpg', '.jpeg'}
        # Glob recursivo por si hay subcarpetas, aunque tu script lo deja plano
        files = sorted([p for p in split_dir.glob("**/*") if p.suffix.lower() in valid_exts])

        if len(files) == 0:
            print(f"⚠️ No se encontraron imágenes en {split_dir}")
            return

        for img_path in files:
            # Parsear nombre: dataset_split_class_id.ext
            # Ej: cable_train_good_000.png
            parts = img_path.stem.split('_')
            
            # Lógica robusta de parsing
            dataset_name = "unknown"
            class_type = "good"
            
            if len(parts) >= 3:
                dataset_name = parts[0]  # 'cable'
                # parts[1] es el split ('train'/'test'/'val')
                # parts[2] suele ser la clase ('good', 'crack', etc.)
                class_type = parts[2]
            
            # 1. Asignar Label Numérico (0-9) para clasificación
            if dataset_name in self.class_names:
                label = self.class_names.index(dataset_name)
            else:
                label = 0
            
            # 2. Flag de Anomalía (0=Normal, 1=Anomalía)
            is_anom = 0 if class_type == "good" else 1
            
            self.image_paths.append(str(img_path))
            self.labels.append(label)
            self.is_anomaly.append(is_anom)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retorna:
            image: Tensor [C, H, W]
            label: Int (índice de la clase del objeto, ej: 0 para cable)
        """
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            image = Image.new('RGB', (128, 128)) # Fallback negro
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        # Retornamos (img, label) para compatibilidad con Lightning
        # Si es un Autoencoder, el training_step ignorará el label.
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
        image_size: int = 128, # Cambiado a int para simplificar config
        **kwargs 
    ):
        super().__init__()
        self.data_dir = data_dir
        self.class_names = class_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # Asegurar tupla
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """
        Carga los datasets 'train', 'validation' y 'test' desde las carpetas
        físicas creadas por el script de procesamiento.
        """
        # Transformaciones
        # Train: Augmentation suave
        train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            # Normalización típica de ImageNet
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        
        # Eval: Solo resize y tensor (sin augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if stage == "fit" or stage is None:
            self.train_dataset = MVTecDataset(
                self.data_dir, 
                split="train", 
                transform=train_transform,
                class_names=self.class_names
            )
            # CARGA EXPLÍCITA DE LA CARPETA VALIDATION
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # Importante: No mezclar para ver imágenes estables en WandB
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_num_samples(self) -> dict:
        return {
            'train': len(self.train_dataset) if self.train_dataset else 0,
            'val': len(self.val_dataset) if self.val_dataset else 0,
            'test': len(self.test_dataset) if self.test_dataset else 0
        }