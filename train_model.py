import os

# --- CORRECCIÓN CRÍTICA PARA CRASH EN WINDOWS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ------------------------------------------------

import torch
import numpy as np
# ... resto de imports ...

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# ==============================================================================
# CALLBACK 1: RECONSTRUCCIÓN DE IMÁGENES (Ya lo tenías)
# ==============================================================================
class ImageReconstructionLogger(pl.Callback):
    """
    Loguea reconstrucciones de imágenes fijas (una por cada clase de MVTec) en WandB.
    Mantiene las mismas imágenes a lo largo de las épocas para ver la evolución.
    """
    def __init__(self, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.val_images = None  # Cache para las imágenes fijas
        # Las 10 clases de MVTec
        self.target_classes = [
            "cable", "capsule", "grid", "hazelnut", "leather", 
            "metal_nut", "pill", "screw", "tile", "transistor"
        ]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Solo ejecutar si es un autoencoder (tiene decoder)
        if not (hasattr(pl_module, 'decoder') or hasattr(pl_module, 'dec_blocks')):
            return

        # 1. Seleccionar imágenes fijas la primera vez (para que sean siempre las mismas)
        if self.val_images is None:
            self.val_images = self._select_diverse_images(trainer, pl_module)
        
        if self.val_images is None:
            return 

        # 2. Reconstruir
        # Mover a dispositivo
        images = self.val_images.to(pl_module.device)
        
        with torch.no_grad():
            pl_module.eval()
            reconstructions = pl_module(images)
            pl_module.train()

        # 3. Loguear
        self._log_images(trainer, images, reconstructions)

    def _select_diverse_images(self, trainer, pl_module):
        """Intenta seleccionar una imagen por cada clase del dataset de validación."""
        try:
            val_loader = trainer.datamodule.val_dataloader()
            dataset = val_loader.dataset
            
            # Manejar Subsets (común cuando se usa random_split)
            indices = list(range(len(dataset)))
            source_dataset = dataset
            if hasattr(dataset, 'indices'):
                indices = dataset.indices
                source_dataset = dataset.dataset
            
            # Buscar una imagen por clase
            selected_indices = []
            found_classes = set()
            
            # Verificar si podemos acceder a los paths (común en MVTecDataset custom)
            if hasattr(source_dataset, 'image_paths'):
                paths = source_dataset.image_paths
                for idx in indices:
                    # Obtener path
                    path = str(paths[idx]).lower()
                    # Verificar a qué clase pertenece
                    for cls in self.target_classes:
                        if cls in path and cls not in found_classes:
                            selected_indices.append(idx)
                            found_classes.add(cls)
                            break
                    # Si ya tenemos las 10, paramos
                    if len(found_classes) >= len(self.target_classes):
                        break
            
            # Si no encontramos suficientes (o el dataset no tiene image_paths accesible)
            # Rellenamos con las primeras disponibles para llegar a num_samples
            if len(selected_indices) < self.num_samples:
                print(f"⚠️ ImageReconstructionLogger: Solo se encontraron clases: {list(found_classes)}")
                for idx in indices:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                    if len(selected_indices) >= self.num_samples:
                        break
            
            # Cargar los tensores de las imágenes seleccionadas
            batch_images = []
            for idx in selected_indices:
                item = source_dataset[idx]
                # El dataset puede devolver (img, label) o solo img
                img = item[0] if isinstance(item, (tuple, list)) else item
                batch_images.append(img)
            
            # Convertir a un solo tensor batch
            return torch.stack(batch_images)

        except Exception as e:
            print(f"⚠️ Error seleccionando imágenes diversas: {e}. Usando batch aleatorio.")
            # Fallback: tomar el primer batch del loader
            batch = next(iter(val_loader))
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            return images[:self.num_samples]

    def _log_images(self, trainer, images, reconstructions):
        images = images.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        
        log_images = []
        for i, (orig, recon) in enumerate(zip(images, reconstructions)):
            # Formato (C, H, W) -> (H, W, C) para visualización
            orig = np.transpose(orig, (1, 2, 0))
            recon = np.transpose(recon, (1, 2, 0))
            
            # Concatenar lado a lado
            combined = np.concatenate((orig, recon), axis=1)
            combined = np.clip(combined, 0, 1)
            
            # Título dinámico
            caption = f"Muestra {i+1}"
            if i < len(self.target_classes):
                # Si logramos ordenarlos, intentamos adivinar el nombre (solo visual)
                # Nota: esto asume que _select_diverse_images encontró en orden, si no, es solo indicativo
                pass 

            log_images.append(wandb.Image(combined, caption=caption))

        trainer.logger.experiment.log({"val/reconstructions": log_images})

# ==============================================================================
# CALLBACK 2: ANÁLISIS DE OVERFITTING (Train vs Val Loss) - CORRECCIÓN 2
# ==============================================================================
class OverfittingCurveCallback(pl.Callback):
    """
    Genera una gráfica comparativa de Train vs Val loss al final del entrenamiento.
    Ayuda a detectar Overfitting visualmente.
    """
    def __init__(self):
        super().__init__()
        self.train_loss_history = []
        self.val_loss_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Intentar obtener métricas logueadas. 
        # Nota: 'train/loss' debe ser logueado con on_epoch=True en el modelo
        metrics = trainer.callback_metrics
        if 'train/loss' in metrics:
            self.train_loss_history.append(metrics['train/loss'].item())
        elif 'train_loss' in metrics:
            self.train_loss_history.append(metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'val/loss' in metrics:
            self.val_loss_history.append(metrics['val/loss'].item())
        elif 'val_loss' in metrics:
            self.val_loss_history.append(metrics['val_loss'].item())

    def on_fit_end(self, trainer, pl_module):
        # Verificar que tengamos datos
        if len(self.train_loss_history) == 0 or len(self.val_loss_history) == 0:
            return

        # Ajustar longitudes (a veces val corre una vez más o menos que train)
        min_len = min(len(self.train_loss_history), len(self.val_loss_history))
        epochs = range(1, min_len + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.train_loss_history[:min_len], 'b-', label='Training Loss')
        ax.plot(epochs, self.val_loss_history[:min_len], 'r-', label='Validation Loss')
        
        ax.set_title(f'Overfitting Analysis: Train vs Val Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Loguear a WandB
        trainer.logger.experiment.log({"analysis/overfitting_curve": wandb.Image(fig)})
        plt.close(fig)
"""
# ==============================================================================
# CALLBACK 3: t-SNE SPACE VISUALIZER 
# ==============================================================================

class TSNEVisualizerCallback(pl.Callback):
    def __init__(self, max_samples=1000):
        super().__init__()
        self.max_samples = max_samples

    def on_test_end(self, trainer, pl_module):
        if not hasattr(pl_module, 'get_embeddings'):
            return

        print("\n🎨 Generando visualización del espacio latente...")
        
        # 1. Configurar Matplotlib para no usar ventana gráfica (evita crashes)
        import matplotlib
        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt
        
        # 2. Recolectar datos
        test_loader = trainer.datamodule.test_dataloader()
        embeddings_list = []
        labels_list = []
        
        pl_module.eval()
        with torch.no_grad():
            for batch in test_loader:
                # Manejo robusto de tuplas
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                    lbls = batch[1]
                else:
                    continue # No hay etiquetas
                
                imgs = imgs.to(pl_module.device)
                emb = pl_module.get_embeddings(imgs)
                
                if len(emb.shape) > 2:
                    emb = torch.flatten(emb, 1)
                
                # Pasar a CPU inmediatamente y limpiar
                embeddings_list.append(emb.cpu().detach().numpy())
                labels_list.append(lbls.cpu().detach().numpy())
                
                if sum([len(x) for x in embeddings_list]) >= self.max_samples:
                    break
        
        if not embeddings_list:
            print("⚠️ No se extrajeron embeddings.")
            return

        # Concatenar
        X = np.concatenate(embeddings_list, axis=0)[:self.max_samples]
        y = np.concatenate(labels_list, axis=0)[:self.max_samples]
        
        # 3. Intentar reducción de dimensionalidad
        df_plot = None
        method_name = ""
        
        try:
            # INTENTO 1: t-SNE (Propenso a fallar en Windows/PyTorch mix)
            from sklearn.manifold import TSNE
            print("   -> Ejecutando t-SNE (n_jobs=1)...")
            
            perp = min(30, len(X) - 1)
            # method='exact' es más lento pero usa menos optimizaciones agresivas que 'barnes-hut'
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', 
                        learning_rate='auto', n_jobs=1, method='barnes_hut')
            
            X_2d = tsne.fit_transform(X)
            method_name = "t-SNE"
            
        except Exception as e:
            print(f"⚠️ t-SNE falló ({e}). Cambiando a PCA...")
            # INTENTO 2: PCA (Muy estable)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            method_name = "PCA"

        # 4. Graficar
        if X_2d is not None:
            try:
                # Crear DataFrame localmente para no depender de pandas externo
                df_data = {'Dim 1': X_2d[:, 0], 'Dim 2': X_2d[:, 1]}
                labels_str = ['Anomaly' if lbl == 1 else 'Good' for lbl in y]
                
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x=df_data['Dim 1'], y=df_data['Dim 2'], 
                                hue=labels_str, style=labels_str,
                                palette={'Good': 'blue', 'Anomaly': 'red'},
                                alpha=0.7)
                
                plt.title(f'{method_name} Latent Space Visualization')
                plt.tight_layout()
                
                trainer.logger.experiment.log({"analysis/latent_space": wandb.Image(plt)})
                plt.close()
                print(f"✅ Visualización ({method_name}) guardada en WandB.")
            except Exception as e:
                print(f"❌ Error al graficar: {e}")
                
"""

# ==============================================================================
# FUNCIÓN DE ENTRENAMIENTO PRINCIPAL
# ==============================================================================

def train_model(cfg: DictConfig, datamodule, model_type: str = "scratch"):
    """
    Entrenar modelo con configuración de Hydra.
    """
    
    # 1. CREAR MODELO
    if model_type == "scratch":
        from src.models.resnet_scratch import ResNetScratchModule
        model = ResNetScratchModule(
            num_classes=cfg.num_classes,
            latent_dim=cfg.model.latent_dim,
            dropout=cfg.model.dropout,
            learning_rate=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            optimizer_config=cfg.optimizer,
            loss_config=cfg.loss,
            max_epochs=cfg.model.max_epochs
        )
        
    elif model_type == "distilled":
        from src.models.resnet_distilled import ResNetDistilledModule
        model = ResNetDistilledModule(
            num_classes=cfg.get('num_classes', getattr(cfg, 'num_classes', 10)),
            latent_dim=cfg.model.get('latent_dim', 128),
            dropout=cfg.model.get('dropout', 0.3),
            learning_rate=cfg.model.get('learning_rate', 1e-3),
            weight_decay=cfg.model.get('weight_decay', 1e-4),
            optimizer_config=cfg.optimizer,
            loss_config=cfg.loss,
            teacher_model=cfg.model.get('teacher_model', 'resnet18'),
            teacher_weights=cfg.model.get('teacher_weights', 'IMAGENET1K_V1'),
            freeze_teacher=cfg.model.get('freeze_teacher', True),
            max_epochs=cfg.get('max_epochs', 100)
        )

    elif model_type == "autoencoder":
        from src.models.autoencoder_unet import UNetAutoencoderModule
        model = UNetAutoencoderModule(
            in_channels=cfg.model.get('in_channels', 3),
            base_channels=cfg.model.get('base_channels', 32),
            depth=cfg.model.get('depth', 4),
            latent_dim=cfg.model.get('latent_dim', 128),
            optimizer_config=cfg.optimizer,
            loss_config=cfg.loss,
            max_epochs=cfg.model.max_epochs
        )
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    
    # 2. CONFIGURAR LOGGER
    if cfg.experiment.run_name:
        run_name = cfg.experiment.run_name
    else:
        model_name = cfg.model.get('name', model_type)
        run_name = f"{model_name}_z{cfg.model.latent_dim}"
    
    print(f"\n Run Name: {run_name}")
    
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=run_name,
        save_dir=cfg.logger.save_dir,
        log_model=cfg.logger.log_model,
        offline=cfg.logger.get('offline', False),
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    
    # 3. CONFIGURAR CALLBACKS (ACTUALIZADO)
    checkpoint_dir = Path(f'./checkpoints/{run_name}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=cfg.callbacks.checkpoint.filename,
            monitor=cfg.callbacks.checkpoint.monitor,
            mode=cfg.callbacks.checkpoint.mode,
            save_top_k=cfg.callbacks.checkpoint.save_top_k,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        
        # --- NUEVOS CALLBACKS SOLICITADOS ---
        OverfittingCurveCallback(),  # Gráfica Train vs Val
        # TSNEVisualizerCallback(max_samples=1000) # t-SNE coloreado por clase
    ]

    # Callback específico solo para Autoencoder (Reconstrucción visual)
    if model_type == "autoencoder":
        callbacks.append(ImageReconstructionLogger(num_samples=4))
    
    
    # 4. TRAINER
    trainer = pl.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.get('log_every_n_steps', 50),
        check_val_every_n_epoch=1,
        deterministic=True
    )
    
    # 5. ENTRENAMIENTO
    print(f"\n=== INICIANDO ENTRENAMIENTO: {run_name} ===")
    trainer.fit(model, datamodule)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    
    # 6. TEST (Importante para generar el t-SNE final)
    print("\n=== EJECUTANDO TEST & T-SNE ===")
    trainer.test(model, datamodule)
    
    wandb.finish()
    
    return trainer, model


# UTILS DE CARGA (Sin cambios mayores, solo soporte de imports)
def load_trained_model(checkpoint_path: str, model_type: str = "scratch"):
    if model_type == "scratch":
        from src.models.resnet_scratch import ResNetScratchModule
        model = ResNetScratchModule.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    elif model_type == "distilled":
        from src.models.resnet_distilled import ResNetDistilledModule
        model = ResNetDistilledModule.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    elif model_type == "autoencoder":
        from src.models.autoencoder_unet import UNetAutoencoderModule
        model = UNetAutoencoderModule.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.eval()
    return model