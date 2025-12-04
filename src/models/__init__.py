"""
Arquitecturas de modelos:
- ResNetCustom: Arquitectura base (conv1-conv3_x de ResNet-18)
- ResNetScratchModule: Lightning module para entrenamiento desde cero
- ResNetDistilledModule: Lightning module con destilación teacher-student
- TeacherModel: ResNet-18 preentrenado como teacher
- DistillationLoss: Loss combinado para destilación
"""

from .resnet_scratch import (
    ResNetScratchModule,
)
from .resnet_distilled import (
    ResNetDistilledModule,
)
from .autoencoder_unet import (
    UNetAutoencoderModule
)

__all__ = [
    # Lightning Modules
    'ResNetScratchModule',
    'ResNetDistilledModule',
]