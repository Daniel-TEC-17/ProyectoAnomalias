import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Any
from omegaconf import DictConfig


class ResNetScratchModule(pl.LightningModule):
    """
    Módulo Lightning para ResNet entrenado desde cero.
    Compatible con configuración completa de Hydra.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 256,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_config: Optional[DictConfig] = None,
        loss_config: Optional[DictConfig] = None,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Construir modelo
        self.model = self._build_resnet(num_classes, latent_dim, dropout)
        
        # Configurar loss
        self._setup_loss(loss_config)
        
        # Guardar para configure_optimizers
        self.optimizer_config = optimizer_config
        self.max_epochs = max_epochs
    
    def _build_resnet(self, num_classes, latent_dim, dropout):
        """Construir el modelo CustomResNet basado en las especificaciones."""
        from src.models.custom_resnet import CustomResNet
        model = CustomResNet(
             num_classes=num_classes,
             latent_dim=latent_dim,
             dropout=dropout
        )
        
         # La referencia a la capa de embedding  se encuentra en CustomResNet
        self.embedding_layer = model.embedding_layer
        return model
    

#Old version 
    """
    def _build_resnet(self, num_classes, latent_dim, dropout):
       
        from torchvision.models import resnet18
        
        # Base ResNet-18 sin pretrained
        model = resnet18(weights=None)
        
        # Modificar clasificador para incluir embedding
        in_features = model.fc.in_features  # 512 para ResNet-18
        
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes)
        )
        
        # Guardar referencia al embedding layer
        self.embedding_layer = model.fc[1]  # La primera Linear layer
        
        return model
    
    """
    def _setup_loss(self, loss_config):
        """Configurar función de pérdida."""
        if loss_config is None or loss_config.name == "cross_entropy":
            label_smoothing = loss_config.get('label_smoothing', 0.0) if loss_config else 0.0
            weight = loss_config.get('weight', None) if loss_config else None
            # Asegurarse de que el peso sea un tensor si se proporciona
            if weight is not None and not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=weight
            )
        else:
            raise ValueError(f"Loss '{loss_config.name}' no soportado para modelo scratch")
    


    def forward(self, x):
        return self.model(x)
    
    def get_embeddings(self, x):
        """
        Extraer embeddings llamando al método del modelo encapsulado.
        """
        return self.model.get_embeddings(x)   
        
    
    def training_step(self, batch, batch_idx):
        # Desempaquetar batch - puede tener 2 o 3 valores
        if len(batch) == 3:
            images, labels, _ = batch  # Ignorar tercer valor (anomaly flag)
        else:
            images, labels = batch
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Métricas
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Desempaquetar batch - puede tener 2 o 3 valores
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Desempaquetar batch - puede tener 2 o 3 valores
        if len(batch) == 3:
            images, labels, is_anomaly = batch
        else:
            images, labels = batch
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test/loss', loss, sync_dist=True)
        self.log('test/acc', acc, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configurar optimizer y scheduler desde config de Hydra.
        """
        cfg = self.optimizer_config
        
        # ============================================================
        # OPTIMIZER
        # ============================================================
        
        if cfg.name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.get('betas', [0.9, 0.999]),
                eps=cfg.get('eps', 1e-8),
                amsgrad=cfg.get('amsgrad', False)
            )
        
        elif cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.get('betas', [0.9, 0.999]),
                eps=cfg.get('eps', 1e-8),
                amsgrad=cfg.get('amsgrad', False)
            )
        
        elif cfg.name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.get('momentum', 0.9),
                weight_decay=cfg.weight_decay,
                nesterov=cfg.get('nesterov', True)
            )
        
        else:
            raise ValueError(f"Optimizer '{cfg.name}' no soportado")
        
        # ============================================================
        # SCHEDULER (si está habilitado)
        # ============================================================
        
        if not cfg.get('scheduler', {}).get('enabled', False):
            return optimizer
        
        sched_cfg = cfg.scheduler
        
        if sched_cfg.name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get('T_max', self.max_epochs),
                eta_min=sched_cfg.get('eta_min', 0)
            )
            return [optimizer], [scheduler]
        
        elif sched_cfg.name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_cfg.get('mode', 'min'),
                factor=sched_cfg.get('factor', 0.5),
                patience=sched_cfg.get('patience', 5),
                threshold=sched_cfg.get('threshold', 1e-4),
                min_lr=sched_cfg.get('min_lr', 1e-6),
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss'
                }
            }
        
        elif sched_cfg.name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get('step_size', 30),
                gamma=sched_cfg.get('gamma', 0.1)
            )
            return [optimizer], [scheduler]
        
        else:
            print(f"⚠️  Scheduler '{sched_cfg.name}' no soportado. Usando solo optimizer.")
            return optimizer
    
    def extract_embeddings_from_dataloader(self, dataloader):
        """
        Extraer embeddings de un DataLoader completo.
        
        Args:
            dataloader: DataLoader con datos
        
        Returns:
            embeddings: numpy array (N, latent_dim)
            labels: numpy array (N,)
        """
        self.eval()
        embeddings_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Desempaquetar batch (puede tener 2 o 3 valores)
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                
                # Extraer embeddings
                emb = self.get_embeddings(images)
                
                embeddings_list.append(emb.cpu())
                labels_list.append(labels)
        
        embeddings = torch.cat(embeddings_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()
        
        return embeddings, labels