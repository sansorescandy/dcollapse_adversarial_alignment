import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

# --------------------------------------------------------
# 1. Función de Selección: Entropía en Destino (Target Entropy)
# --------------------------------------------------------
def calculate_target_entropy(model, target_loader, device):
    """
    Calcula la entropía media de las predicciones en el dominio destino.
    Un valor menor indica que el modelo está más "seguro" de sus predicciones.
    """
    model.eval()
    total_entropy = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, _ in target_loader:
            x = x.to(device)
            # Solo nos interesan los logits de clase, no el discriminador
            class_logits, _, _ = model(x, alpha=0.0) 
            
            # Convertir logits a probabilidades
            probs = nn.Softmax(dim=1)(class_logits)
            
            # H(p) = - sum(p * log(p + epsilon))
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1)
            
            total_entropy += torch.sum(entropy).item()
            total_samples += x.size(0)
            
    return total_entropy / total_samples

# --------------------------------------------------------
# 2. Ajuste Dinámico de Parámetros (LR y Alpha)
# --------------------------------------------------------
def adjust_learning_rate(optimizer, epoch, num_epochs, base_lr=0.01):
    """
    Aplica el decaimiento polinomial a la tasa de aprendizaje.
    """
    p = epoch / num_epochs
    lr = base_lr / math.pow((1 + 10 * p), 0.75)
    
    # El optimizador tendrá 3 grupos de parámetros: backbone (lr*0.1), clasificador (lr), discriminador (lr)
    optimizer.param_groups[0]['lr'] = lr * 0.1
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    return lr

def calculate_alpha(epoch, num_epochs):
    """
    Calcula el peso adversarial (alpha) para el GRL usando una curva logística.
    """
    p = epoch / num_epochs
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

# --------------------------------------------------------
# 3. Bucle de Entrenamiento Principal
# --------------------------------------------------------
def train_dann(model, source_loader, target_loader, num_epochs, device, base_lr=0.01):
    """
    Orquesta el entrenamiento de la red adversarial y guarda el mejor modelo basado en Target Entropy.
    """
    model = model.to(device)
    
    # Criterios de pérdida
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    # Grupos de parámetros con LR diferencial
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': base_lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': base_lr},
        {'params': model.discriminator.parameters(), 'lr': base_lr}
    ], momentum=0.9, weight_decay=5e-4)

    best_entropy = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        
        # Ajustar hiperparámetros dinámicos
        current_lr = adjust_learning_rate(optimizer, epoch, num_epochs, base_lr)
        current_alpha = calculate_alpha(epoch, num_epochs)
        
        # Iteradores
        iter_source = iter(source_loader)
        iter_target = iter(target_loader)
        num_batches = min(len(source_loader), len(target_loader))
        
        epoch_loss_cls = 0.0
        epoch_loss_dom = 0.0
        
        for _ in range(num_batches):
            # Cargar datos
            data_source, label_source = next(iter_source)
            data_target = next(iter_target)
            # Desempaquetar el target si el loader devuelve (imagen, etiqueta_falsa)
            if isinstance(data_target, list) or isinstance(data_target, tuple):
                data_target = data_target[0]
            
            data_source, label_source = data_source.to(device), label_source.to(device)
            data_target = data_target.to(device)
            
            optimizer.zero_grad()
            
            # --- ENTRENAMIENTO CON DOMINIO FUENTE ---
            # Etiquetas de dominio: Fuente = 0
            domain_label_source = torch.zeros(data_source.size(0), dtype=torch.long).to(device)
            
            class_logits_s, domain_logits_s, _ = model(data_source, alpha=current_alpha)
            
            loss_cls = criterion_class(class_logits_s, label_source)
            loss_dom_s = criterion_domain(domain_logits_s, domain_label_source)
            
            # --- ENTRENAMIENTO CON DOMINIO DESTINO ---
            # Etiquetas de dominio: Destino = 1
            domain_label_target = torch.ones(data_target.size(0), dtype=torch.long).to(device)
            
            _, domain_logits_t, _ = model(data_target, alpha=current_alpha)
            loss_dom_t = criterion_domain(domain_logits_t, domain_label_target)
            
            # --- PÉRDIDA TOTAL ---
            loss_dom_total = loss_dom_s + loss_dom_t
            loss_total = loss_cls + loss_dom_total
            
            loss_total.backward()
            optimizer.step()
            
            epoch_loss_cls += loss_cls.item()
            epoch_loss_dom += loss_dom_total.item()

        # Evaluación no supervisada al final de la época
        target_entropy = calculate_target_entropy(model, target_loader, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss Cls: {epoch_loss_cls/num_batches:.4f} | "
              f"Loss Dom: {epoch_loss_dom/num_batches:.4f} | "
              f"Alpha: {current_alpha:.3f} | "
              f"Target Entropy: {target_entropy:.4f}")
        
        # Selección del modelo: Guardamos el que tenga menor Target Entropy
        if target_entropy < best_entropy:
            best_entropy = target_entropy
            best_model_state = model.state_dict()
            print(f"  -> Nuevo mejor modelo encontrado! (Entropía: {best_entropy:.4f})")

    print("Entrenamiento finalizado.")
    # Restaurar los pesos del mejor modelo encontrado
    model.load_state_dict(best_model_state)
    return model
