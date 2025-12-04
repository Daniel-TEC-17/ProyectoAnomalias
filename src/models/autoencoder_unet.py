import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, List, Tuple
from omegaconf import DictConfig

# --- Bloques Constructivos (Portados del Notebook) ---

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        in_ch: número de canales de entrada al bloque (antes del upsample)
        out_ch: número de canales de salida deseados después del bloque
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # Se concatena con skip (out_ch + skip_ch), asumimos skip_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Ajuste de tamaño por si hay diferencias de redondeo en dimensiones impares
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# --- Utilidades para Loss (SSIM) ---

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, channel=3, K1=0.01, K2=0.03):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = channel
        self.K1 = K1
        self.K2 = K2
        self.register_buffer("window", self._create_window(window_size, sigma, channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.tensor([(-(x - window_size // 2) ** 2) / float(2 * sigma**2) for x in range(window_size)])
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma, channel):
        _1d_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2d_window = _1d_window @ _1d_window.t()
        _2d_window = _2d_window.float().unsqueeze(0).unsqueeze(0)
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, x, y):
        if x.size(1) != self.channel or y.size(1) != self.channel:
            self.channel = x.size(1)
            self.window = self._create_window(self.window_size, self.sigma, self.channel).to(x.device)

        C1 = (self.K1 * self.data_range) ** 2
        C2 = (self.K2 * self.data_range) ** 2

        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=self.channel)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, self.window, padding=self.window_size // 2, groups=self.channel) - mu_x2
        sigma_y2 = F.conv2d(y * y, self.window, padding=self.window_size // 2, groups=self.channel) - mu_y2
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size // 2, groups=self.channel) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = num / (den + 1e-8)
        return 1 - ssim_map.mean()


# --- Módulo Principal ---

class UNetAutoencoderModule(pl.LightningModule):
    """
    Módulo Lightning para U-Net Autoencoder.
    Adaptado para ser instanciado via Hydra con estructura similar a ResNetDistilled.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        latent_dim: int = 128,  # Usado para el bottleneck
        optimizer_config: Optional[DictConfig] = None,
        loss_config: Optional[DictConfig] = None,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()

        # Construir arquitectura U-Net
        self._build_model(in_channels, base_channels, depth)

        # Configurar Loss
        self._setup_loss(loss_config)

        # Guardar configs
        self.optimizer_config = optimizer_config
        self.max_epochs = max_epochs

    def _build_model(self, in_ch, base, depth):
        """Construye las capas Encoder, Bottleneck y Decoder."""
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        ch = in_ch
        
        # Encoder
        for d in range(depth):
            out_ch = base * (2 ** d)
            self.enc_blocks.append(UNetEncoderBlock(ch, out_ch))
            ch = out_ch
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2))

        # Bottleneck
        self.bottleneck = UNetEncoderBlock(ch, ch * 2)
        ch = ch * 2

        # Decoder
        self.dec_blocks = nn.ModuleList()
        # Construimos decoder en reverso
        for d in reversed(range(depth)):
            out_ch = base * (2 ** d)
            self.dec_blocks.append(UNetDecoderBlock(ch, out_ch))
            ch = out_ch

        # Output final
        self.final_conv = nn.Conv2d(base, in_ch, kernel_size=1)
        self.activation = nn.Sigmoid()

    def _setup_loss(self, loss_config):
        """Configura la función de pérdida basada en el config de Hydra."""
        if loss_config is None:
            # Default fallback
            self.criterion = nn.MSELoss()
            return

        loss_type = loss_config.get('type', 'MSELoss')
        
        if loss_type == "L1Loss":
            self.criterion = nn.L1Loss()
        elif loss_type == "MSELoss":
            self.criterion = nn.MSELoss()
        elif loss_type == "SSIM":
            self.criterion = SSIMLoss(
                window_size=loss_config.get('window_size', 11),
                sigma=loss_config.get('sigma', 1.5),
                data_range=loss_config.get('data_range', 1.0)
            )
        elif loss_type == "SSIM_L1":
            # Combinación lineal wrapper
            class SSIML1Combined(nn.Module):
                def __init__(self, ssim_loss, l1_weight):
                    super().__init__()
                    self.ssim = ssim_loss
                    self.l1 = nn.L1Loss()
                    self.l1_weight = l1_weight
                
                def forward(self, x, y):
                    return self.ssim(x, y) + self.l1_weight * self.l1(x, y)
            
            ssim_part = SSIMLoss(
                window_size=loss_config.get('window_size', 11),
                sigma=loss_config.get('sigma', 1.5),
                data_range=loss_config.get('data_range', 1.0)
            )
            self.criterion = SSIML1Combined(ssim_part, loss_config.get('l1_weight', 0.1))
        else:
            raise ValueError(f"Loss type '{loss_type}' no soportado en UNetAutoencoder")

    def forward(self, x):
        """Forward pass de la U-Net."""
        skips = []
        out = x

        # Encoder path
        for enc, down in zip(self.enc_blocks, self.downsamples):
            out = enc(out)
            skips.append(out)
            out = down(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder path
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            out = dec(out, skip)

        out = self.final_conv(out)
        return self.activation(out)

    def _shared_step(self, batch, stage):
        """Lógica compartida para train/val/test."""
        # Manejo flexible de tuplas de batch (img, label) o solo (img)
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def get_embeddings(self, x):
        """
        Extrae los embeddings del bottleneck.
        Útil para visualización (PCA/t-SNE) o detección de anomalías.
        """
        # 1. Pasar por el Encoder (igual que en forward)
        out = x
        for enc, down in zip(self.enc_blocks, self.downsamples):
            out = enc(out)
            out = down(out)

        # 2. Pasar por el Bottleneck
        out = self.bottleneck(out)
        
        # 3. Aplanar: (Batch, Canales, H, W) -> (Batch, Vector)
        # Esto convierte el mapa de características 3D en un vector de embeddings 1D
        embeddings = torch.flatten(out, 1)
        
        return embeddings
    
    def configure_optimizers(self):
        """Configuración robusta del optimizador (copiada del estilo de ResNetDistilled)."""
        cfg = self.optimizer_config
        if cfg is None:
            # Fallback simple
            return torch.optim.Adam(self.parameters(), lr=1e-3)

        params = self.parameters()

        # Selección de Optimizador
        if cfg.name == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.get('weight_decay', 0.0),
                betas=cfg.get('betas', [0.9, 0.999])
            )
        elif cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.get('weight_decay', 0.01)
            )
        elif cfg.name == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=cfg.learning_rate,
                momentum=cfg.get('momentum', 0.9),
                weight_decay=cfg.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Optimizer '{cfg.name}' no soportado")

        # Configuración de Scheduler (Opcional)
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
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss'
                }
            }
        
        return optimizer