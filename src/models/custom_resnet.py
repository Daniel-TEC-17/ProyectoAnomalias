import torch.nn as nn
from torchvision.models import resnet18

class CustomResNet(nn.Module):
    """
    Una arquitectura ResNet personalizada que utiliza solo los primeros 3 bloques
    convolucionales de un ResNet-18 y un clasificador personalizado.

    Esto sigue la especificación del documento del proyecto.
    """
    def __init__(self, num_classes: int, latent_dim: int, dropout: float):
        super().__init__()
        
        # Cargar un ResNet-18 base, no necesitamos los pesos pre-entrenados aquí.
        base_model = resnet18(weights=None)
        
        # 1. Extraer las capas convolucionales requeridas (hasta layer3)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3  # Salida de 256 canales para ResNet-18
        )
        
        # 2. Definir un pool y un clasificador personalizado
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # El clasificador es idéntico al que ya usabas, pero ahora se aplica
        # sobre una entrada de 256 canales en lugar de 512.
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim), # <-- La entrada es 256
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes)
        )
        
        # Guardar referencia a la capa de embedding para fácil acceso
        self.embedding_layer = self.fc[1]

    def forward(self, x):
        """Define el forward pass."""
        # Extraer features de las capas convolucionales
        x = self.features(x)
        
        # Pooling y flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Clasificación
        x = self.fc(x)
        
        return x

    def get_embeddings(self, x):
        """
        Método para extraer los embeddings (la salida de la primera capa lineal
        en el clasificador).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Aplicar solo la primera parte del clasificador
        x = self.fc[0](x) # Dropout
        embeddings = self.fc[1](x) # Linear -> embedding
        
        return embeddings
