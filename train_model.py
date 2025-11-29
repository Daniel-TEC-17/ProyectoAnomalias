import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path


def train_model(cfg: DictConfig, datamodule, model_type: str = "scratch"):
    """
    Entrenar modelo con configuración de Hydra.
    
    Args:
        cfg: Configuración completa de Hydra
        datamodule: DataModule de PyTorch Lightning
        model_type: Tipo de modelo ('scratch' o 'distilled')
    
    Returns:
        trainer: Entrenador de PyTorch Lightning
        model: Modelo entrenado
    """
    
    # ==================================================================
    # 1. CREAR MODELO SEGÚN TIPO
    # ==================================================================
    
    if model_type == "scratch":
        from models.resnet_scratch import ResNetScratchModule
        
        model = ResNetScratchModule(
            num_classes=cfg.num_classes,
            latent_dim=cfg.model.latent_dim,
            dropout=cfg.model.dropout,
            learning_rate=cfg.optimizer.learning_rate,  # Desde optimizer
            weight_decay=cfg.optimizer.weight_decay,
            # Pasar toda la config de optimizer
            optimizer_config=cfg.optimizer,
            # Pasar config de loss
            loss_config=cfg.loss,
            max_epochs=cfg.model.max_epochs
        )
        
    elif model_type == "distilled":
        from models.resnet_distilled import ResNetDistilledModule
        
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'scratch' or 'distilled'")
    
    
    # ==================================================================
    # 2. GENERAR NOMBRE DEL RUN
    # ==================================================================
    
    if cfg.experiment.run_name:
        run_name = cfg.experiment.run_name
    else:
        # Generar automáticamente
        model_name = cfg.model.get('name', model_type)
        run_name = (f"{model_name}_"
                   f"z{cfg.model.latent_dim}_"
                   f"lr{cfg.optimizer.learning_rate}_"
                   f"bs{cfg.data.batch_size}")
    
    print(f"\n Run Name: {run_name}")
    
    
    # ==================================================================
    # 3. CONFIGURAR LOGGER (WandB)
    # ==================================================================
    
    # Preparar tags
    tags = cfg.logger.get('tags', []).copy() if cfg.logger.get('tags') else []
    tags.extend([
        model_type,
        f"z{cfg.model.latent_dim}",
        cfg.optimizer.name
    ])
    
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=run_name,
        save_dir=cfg.logger.save_dir,
        log_model=cfg.logger.log_model,
        offline=cfg.logger.get('offline', False),
        tags=tags,
        notes=cfg.logger.get('notes', None),
        entity=cfg.logger.get('entity', None)
    )
    
    # Log de hiperparámetros completos
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    
    # ==================================================================
    # 4. CONFIGURAR CALLBACKS
    # ==================================================================
    
    checkpoint_dir = Path(f'./checkpoints/{run_name}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
            min_delta=cfg.callbacks.early_stopping.min_delta,
            verbose=True
        ),
        
        # Model Checkpoint
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=cfg.callbacks.checkpoint.filename,
            monitor=cfg.callbacks.checkpoint.monitor,
            mode=cfg.callbacks.checkpoint.mode,
            save_top_k=cfg.callbacks.checkpoint.save_top_k,
            save_last=cfg.callbacks.checkpoint.save_last,
            verbose=True
        ),
        
        # Learning Rate Monitor
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    
    # ==================================================================
    # 5. CONFIGURAR TRAINER
    # ==================================================================
    
    trainer = pl.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.get('log_every_n_steps', 50),
        check_val_every_n_epoch=cfg.trainer.get('check_val_every_n_epoch', 1),
        deterministic=cfg.trainer.get('deterministic', True),
        gradient_clip_val=cfg.trainer.get('gradient_clip_val', 1.0)
    )
    
    
    # ==================================================================
    # 6. IMPRIMIR RESUMEN
    # ==================================================================
    
    print("\n" + "="*80)
    print(f" INICIANDO ENTRENAMIENTO: {run_name}")
    print("="*80)
    print(f" Modelo: {cfg.model.get('name', model_type)}")
    print(f" Latent Dim: {cfg.model.latent_dim}")
    print(f" Learning Rate: {cfg.optimizer.learning_rate}")
    print(f"  Optimizer: {cfg.optimizer.name}")
    
    # Scheduler info
    if cfg.optimizer.get('scheduler', {}).get('enabled', False):
        print(f" Scheduler: {cfg.optimizer.scheduler.name}")
    else:
        print(f" Scheduler: None")
    
    print(f" Max Epochs: {cfg.model.max_epochs}")
    print(f" Batch Size: {cfg.data.batch_size}")
    
    # Loss info
    if model_type == "distilled":
        print(f" Loss: {cfg.loss.name} (T={cfg.loss.temperature}, α={cfg.loss.alpha}, β={cfg.loss.beta})")
    else:
        print(f" Loss: {cfg.loss.name}")
    
    print(f" Checkpoint Dir: {checkpoint_dir}")
    print("="*80 + "\n")
    
    
    # ==================================================================
    # 7. ENTRENAR
    # ==================================================================
    
    trainer.fit(model, datamodule)
    
    
    # ==================================================================
    # 8. FINALIZAR Y REPORTAR
    # ==================================================================
    
    print("\n" + "="*80)
    print(" ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f" Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f" Best val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("="*80 + "\n")
    
    wandb.finish()
    
    return trainer, model


def load_trained_model(checkpoint_path: str, model_type: str = "scratch"):
    """
    Cargar modelo desde checkpoint.
    
    Args:
        checkpoint_path: Ruta al checkpoint (.ckpt)
        model_type: Tipo de modelo ('scratch' o 'distilled')
    
    Returns:
        model: Modelo cargado en modo eval
    """
    if model_type == "scratch":
        from models.resnet_scratch import ResNetScratchModule
        model = ResNetScratchModule.load_from_checkpoint(checkpoint_path)
    elif model_type == "distilled":
        from models.resnet_distilled import ResNetDistilledModule
        model = ResNetDistilledModule.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.eval()
    print(f" Modelo cargado desde: {checkpoint_path}")
    return model


def get_best_checkpoint_path(run_name: str):
    """
    Obtener la ruta del mejor checkpoint de un run.
    
    Args:
        run_name: Nombre del run
    
    Returns:
        best_ckpt_path: Ruta al mejor checkpoint
    """
    checkpoint_dir = Path(f'./checkpoints/{run_name}')
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No se encontró directorio: {checkpoint_dir}")
    
    # Buscar archivo 'best-*.ckpt'
    best_ckpts = list(checkpoint_dir.glob('best-*.ckpt'))
    
    if not best_ckpts:
        raise FileNotFoundError(f"No se encontró 'best-*.ckpt' en {checkpoint_dir}")
    
    # Tomar el más reciente
    best_ckpt_path = max(best_ckpts, key=lambda p: p.stat().st_mtime)
    
    return str(best_ckpt_path)


def get_last_checkpoint_path(run_name: str):
    """
    Obtener la ruta del último checkpoint de un run.
    
    Args:
        run_name: Nombre del run
    
    Returns:
        last_ckpt_path: Ruta al último checkpoint
    """
    checkpoint_dir = Path(f'./checkpoints/{run_name}')
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No se encontró directorio: {checkpoint_dir}")
    
    last_ckpt = checkpoint_dir / 'last.ckpt'
    
    if not last_ckpt.exists():
        raise FileNotFoundError(f"No se encontró 'last.ckpt' en {checkpoint_dir}")
    
    return str(last_ckpt)