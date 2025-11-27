import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional
from omegaconf import DictConfig


class ResNetDistilledModule(pl.LightningModule):
    """
    Módulo Lightning para ResNet con Knowledge Distillation.
    Usa un teacher pretrained y entrena un student desde cero.
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
        teacher_model: str = "resnet18",
        teacher_weights: str = "IMAGENET1K_V1",
        freeze_teacher: bool = True,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Configurar teacher
        self.teacher = self._build_teacher(teacher_model, teacher_weights, num_classes, freeze_teacher)
        
        # Configurar student (misma arquitectura que scratch)
        self.student = self._build_student(num_classes, latent_dim, dropout)
        
        # Configurar loss
        self._setup_loss(loss_config)
        
        # Guardar configs
        self.optimizer_config = optimizer_config
        self.max_epochs = max_epochs
    
    def _build_teacher(self, model_name, weights, num_classes, freeze):
        """Construir modelo teacher."""
        from torchvision.models import resnet18, ResNet18_Weights
        
        if model_name == "resnet18":
            # Cargar con pesos pretrained
            if weights == "IMAGENET1K_V1":
                teacher = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                teacher = resnet18(weights=None)
            
            # Adaptar última capa a num_classes
            in_features = teacher.fc.in_features
            teacher.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Teacher model '{model_name}' no soportado")
        
        # Congelar si es necesario
        if freeze:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
        
        return teacher
    
    def _build_student(self, num_classes, latent_dim, dropout):
        """Construir modelo student."""
        from torchvision.models import resnet18
        
        student = resnet18(weights=None)
        
        in_features = student.fc.in_features
        student.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes)
        )
        
        self.embedding_layer = student.fc[1]
        
        return student
    
    def _setup_loss(self, loss_config):
        """Configurar función de pérdida para distillation."""
        if loss_config is None or loss_config.type != "teacher_student":
            raise ValueError("loss_config debe ser de tipo 'teacher_student'")
        
        self.temperature = loss_config.get('temperature', 5.0)
        self.alpha = loss_config.get('alpha', 0.3)  # Hard loss
        self.beta = loss_config.get('beta', 0.5)    # Soft loss
        self.gamma = loss_config.get('gamma', 0.2)   # Feature loss
        
        # Verificar que sumen aprox 1
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 0.01:
            print(f"⚠️  Warning: alpha + beta + gamma = {total} (debería ser ~1.0)")
        
        # Hard loss (Cross Entropy con labels)
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Soft loss (KL Divergence entre teacher y student)
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Feature loss (MSE entre embeddings)
        self.feature_loss = nn.MSELoss()
    
    def forward(self, x):
        """Forward del student."""
        return self.student(x)
    
    def get_embeddings(self, x):
        """Extraer embeddings del student."""
        x = self.student.conv1(x)
        x = self.student.bn1(x)
        x = self.student.relu(x)
        x = self.student.maxpool(x)
        
        x = self.student.layer1(x)
        x = self.student.layer2(x)
        x = self.student.layer3(x)
        x = self.student.layer4(x)
        
        x = self.student.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.student.fc[0](x)  # Dropout
        embeddings = self.student.fc[1](x)  # Linear -> embedding
        
        return embeddings
    
    def _get_teacher_embeddings(self, x):
        """Extraer embeddings del teacher."""
        x = self.teacher.conv1(x)
        x = self.teacher.bn1(x)
        x = self.teacher.relu(x)
        x = self.teacher.maxpool(x)
        
        x = self.teacher.layer1(x)
        x = self.teacher.layer2(x)
        x = self.teacher.layer3(x)
        x = self.teacher.layer4(x)
        
        x = self.teacher.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x  # Embedding antes de FC
    
    def compute_distillation_loss(self, student_logits, teacher_logits, 
                                   student_embeddings, teacher_embeddings, labels):
        """
        Calcular loss de distillation combinado.
        
        Returns:
            total_loss: pérdida total
            losses_dict: diccionario con componentes individuales
        """
        # 1. Hard Loss (con ground truth labels)
        loss_hard = self.hard_loss(student_logits, labels)
        
        # 2. Soft Loss (KL divergence entre outputs suavizados)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        loss_soft = self.soft_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 3. Feature Loss (MSE entre embeddings)
        loss_feature = self.feature_loss(student_embeddings, teacher_embeddings)
        
        # Loss total ponderado
        total_loss = (self.alpha * loss_hard + 
                     self.beta * loss_soft + 
                     self.gamma * loss_feature)
        
        losses_dict = {
            'hard': loss_hard.item(),
            'soft': loss_soft.item(),
            'feature': loss_feature.item(),
            'total': total_loss.item()
        }
        
        return total_loss, losses_dict
    
    def training_step(self, batch, batch_idx):
        # Desempaquetar batch - puede tener 2 o 3 valores
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        # Forward student
        student_logits = self.student(images)
        student_embeddings = self.get_embeddings(images)
        
        # Forward teacher (sin gradientes)
        with torch.no_grad():
            teacher_logits = self.teacher(images)
            teacher_embeddings = self._get_teacher_embeddings(images)
        
        # Calcular loss
        total_loss, losses_dict = self.compute_distillation_loss(
            student_logits, teacher_logits,
            student_embeddings, teacher_embeddings,
            labels
        )
        
        # Métricas
        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_hard', losses_dict['hard'], on_step=False, on_epoch=True)
        self.log('train/loss_soft', losses_dict['soft'], on_step=False, on_epoch=True)
        self.log('train/loss_feature', losses_dict['feature'], on_step=False, on_epoch=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Desempaquetar batch - puede tener 2 o 3 valores
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        student_logits = self.student(images)
        student_embeddings = self.get_embeddings(images)
        
        with torch.no_grad():
            teacher_logits = self.teacher(images)
            teacher_embeddings = self._get_teacher_embeddings(images)
        
        total_loss, losses_dict = self.compute_distillation_loss(
            student_logits, teacher_logits,
            student_embeddings, teacher_embeddings,
            labels
        )
        
        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/loss_hard', losses_dict['hard'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_soft', losses_dict['soft'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_feature', losses_dict['feature'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        # Desempaquetar batch - puede tener 2 o 3 valores
        if len(batch) == 3:
            images, labels, is_anomaly = batch
        else:
            images, labels = batch
        
        student_logits = self.student(images)
        
        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test/acc', acc, sync_dist=True)
        return acc
    
    def configure_optimizers(self):
        """Configurar optimizer y scheduler (solo para student)."""
        cfg = self.optimizer_config
        
        # Solo optimizar parámetros del student
        if cfg.name == 'adam':
            optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.get('betas', [0.9, 0.999]),
                eps=cfg.get('eps', 1e-8),
                amsgrad=cfg.get('amsgrad', False)
            )
        
        elif cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.get('betas', [0.9, 0.999]),
                eps=cfg.get('eps', 1e-8),
                amsgrad=cfg.get('amsgrad', False)
            )
        
        elif cfg.name == 'sgd':
            optimizer = torch.optim.SGD(
                self.student.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.get('momentum', 0.9),
                weight_decay=cfg.weight_decay,
                nesterov=cfg.get('nesterov', True)
            )
        
        else:
            raise ValueError(f"Optimizer '{cfg.name}' no soportado")
        
        # Scheduler
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
            print(f"⚠️  Scheduler '{sched_cfg.name}' no soportado.")
            return optimizer
    
    def extract_embeddings_from_dataloader(self, dataloader):
        """Extraer embeddings del student."""
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
                emb = self.get_embeddings(images)
                embeddings_list.append(emb.cpu())
                labels_list.append(labels)
        
        embeddings = torch.cat(embeddings_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()
        
        return embeddings, labels