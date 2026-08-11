import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function

# --------------------------------------------------------
# 1. Gradient Reversal Layer (GRL)
# --------------------------------------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# --------------------------------------------------------
# 2. Backbone + Bottleneck
# --------------------------------------------------------
class ResNet50Backbone(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super(ResNet50Backbone, self).__init__()
        # Cargar ResNet-50 pre-entrenado en ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Extraer todas las capas excepto la última FC y el pooling
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Bottleneck: 2048 -> 256 (como dicta el protocolo UDA)
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        # Inicialización de Xavier para el bottleneck
        nn.init.xavier_uniform_(self.bottleneck[0].weight, gain=1.0)
        nn.init.constant_(self.bottleneck[0].bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        return x # Este es nuestro feature vector de 256-d para el FFI

# --------------------------------------------------------
# 3. Clasificador de Tarea
# --------------------------------------------------------
class TaskClassifier(nn.Module):
    def __init__(self, in_features=256, num_classes=31): # 31 para Office-31
        super(TaskClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)

# --------------------------------------------------------
# 4. Discriminador de Dominio (DANN)
# --------------------------------------------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=256):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),  # salida binaria para BCELoss
            nn.Sigmoid()
        )
        
        # Inicialización
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha):
        x = GradientReversalFunction.apply(x, alpha)
        return self.net(x)

# --------------------------------------------------------
# 5. Modelo DANN Completo (Orquestador)
# --------------------------------------------------------
class DANNModel(nn.Module):
    def __init__(self, num_classes=31):
        super(DANNModel, self).__init__()
        self.backbone = ResNet50Backbone(bottleneck_dim=256)
        self.classifier = TaskClassifier(in_features=256, num_classes=num_classes)
        self.discriminator = DomainDiscriminator(in_features=256)

    def forward(self, x, alpha=1.0):
        # 1. Extraer características
        features = self.backbone(x)
        
        # 2. Predicción de clase (Tarea principal)
        class_output = self.classifier(features)
        
        # 3. Predicción de dominio (Rama adversarial)
        domain_output = self.discriminator(features, alpha)
        
        return class_output, domain_output, features

# --------------------------------------------------------
# 6. Proyección Aleatoria para CDAN (Random Multilinear)
# --------------------------------------------------------
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim_list[i], output_dim), requires_grad=False)
            for i in range(self.input_num)
        ])

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

# --------------------------------------------------------
# 7. Modelo CDAN Completo
# --------------------------------------------------------
class CDANModel(nn.Module):
    def __init__(self, num_classes=31):
        super(CDANModel, self).__init__()
        # Reutilizamos las mismas piezas arquitectónicas del DANN
        self.backbone = ResNet50Backbone(bottleneck_dim=256)
        self.classifier = TaskClassifier(in_features=256, num_classes=num_classes)
        
        # El discriminador ahora recibe 1024 dimensiones (fruto de la proyección aleatoria)
        self.discriminator = DomainDiscriminator(in_features=1024)
        
        # Capa de proyección: recibe características (256) y predicciones softmax (31)
        self.random_layer = RandomLayer([256, num_classes], output_dim=1024)

    def forward(self, x, alpha=1.0):
        # 1. Extraer características (256-d)
        features = self.backbone(x)
        
        # 2. Predicción de clase (logits)
        class_logits = self.classifier(features)
        
        # 3. Aplicar Softmax para obtener probabilidades (necesarias para el condicionamiento)
        softmax_output = nn.Softmax(dim=1)(class_logits)
        
        # 4. Producto tensorial aleatorio: f ⊗ g -> proyección a 1024-d
        # Se desprende el gradiente de las predicciones para estabilizar el entrenamiento
        random_out = self.random_layer([features, softmax_output.detach()])
        
        # 5. Predicción de dominio usando el vector condicionado
        domain_output = self.discriminator(random_out, alpha)
        
        return class_logits, domain_output, features
