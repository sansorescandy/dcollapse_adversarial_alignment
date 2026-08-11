import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_target_entropy(model, dataloader, device):
    """Calcula la entropía de Shannon en el dominio destino (sin etiquetas)."""
    model.eval()
    entropies = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            # Extraemos los logits, ignoramos logits de dominio y features
            class_logits, _, _ = model(x, alpha=0.0)
            
            # Cálculo de la entropía
            probs = F.softmax(class_logits, dim=1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            entropies.append(batch_entropy.item())
            
    return sum(entropies) / len(entropies)

def evaluate_source_domain(model, dataloader, device):
    """Evalúa la pérdida de clasificación y precisión en el dominio origen."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            class_logits, _, _ = model(x, alpha=0.0)
            
            loss = criterion(class_logits, y)
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(class_logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def train_dann(model, train_source, val_source, target_loader, num_epochs, device, method, source_name, target_name, seed):
    """
    Bucle de entrenamiento UDA optimizado.
    Soporta DANN y CDAN, integrando registro de métricas y selección por entropía.
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCELoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    # Inicializar el diccionario para exportar a CSV
    history = {
        'epoch': [], 'train_loss_cls': [], 'train_loss_domain': [],
        'val_source_loss': [], 'val_source_acc': [], 
        'val_target_entropy': []
    }
    
    best_target_entropy = float('inf')
    best_model_path = f"./checkpoints/best_model_{method}_{source_name}_to_{target_name}_seed{seed}.pth"
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss_cls = 0.0
        epoch_loss_domain = 0.0
        total_samples = 0
        
        # Cálculo dinámico del parámetro alpha para el GRL
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + float(np.exp(-10. * p))) - 1
        
        # Iterador conjunto para equilibrar lotes de origen y destino
        len_dataloader = min(len(train_source), len(target_loader))
        source_iter = iter(train_source)
        target_iter = iter(target_loader)
        
        for i in range(len_dataloader):
            # 1. Datos origen
            s_img, s_label = next(source_iter)
            s_img, s_label = s_img.to(device), s_label.to(device)
            
            # 2. Datos destino
            t_img, _ = next(target_iter)
            t_img = t_img.to(device)
            
            optimizer.zero_grad()
            
            # --- Forward Pass (Origen) ---
            class_logits, domain_logits_s, _ = model(s_img, alpha=alpha)
            loss_cls = criterion_cls(class_logits, s_label)
            
            # Las etiquetas del clasificador de dominio para origen son 0
            domain_label_s = torch.zeros(domain_logits_s.size(0), 1).to(device)
            loss_domain_s = criterion_domain(domain_logits_s, domain_label_s)
            
            # --- Forward Pass (Destino) ---
            _, domain_logits_t, _ = model(t_img, alpha=alpha)
            
            # Las etiquetas del clasificador de dominio para destino son 1
            domain_label_t = torch.ones(domain_logits_t.size(0), 1).to(device)
            loss_domain_t = criterion_domain(domain_logits_t, domain_label_t)
            
            # --- Backpropagation ---
            loss_domain = loss_domain_s + loss_domain_t
            loss = loss_cls + loss_domain
            loss.backward()
            optimizer.step()
            
            epoch_loss_cls += loss_cls.item() * s_img.size(0)
            epoch_loss_domain += loss_domain.item() * s_img.size(0)
            total_samples += s_img.size(0)
            
        avg_loss_cls = epoch_loss_cls / total_samples
        avg_loss_domain = epoch_loss_domain / total_samples
        
        # --- Fase de Validación al finalizar la época ---
        val_src_loss, val_src_acc = evaluate_source_domain(model, val_source, device)
        val_tgt_entropy = calculate_target_entropy(model, target_loader, device)
        
        # Registrar métricas
        history['epoch'].append(epoch)
        history['train_loss_cls'].append(avg_loss_cls)
        history['train_loss_domain'].append(avg_loss_domain)
        history['val_source_loss'].append(val_src_loss)
        history['val_source_acc'].append(val_src_acc)
        history['val_target_entropy'].append(val_tgt_entropy)
        
        print(f"Epoch [{epoch}/{num_epochs}] | Cls Loss: {avg_loss_cls:.4f} | Src Acc: {val_src_acc:.2f}% | Tgt Entropy: {val_tgt_entropy:.4f}")
        
        # Selección de Modelo No Supervisada (Mínima Entropía)
        if val_tgt_entropy < best_target_entropy:
            best_target_entropy = val_tgt_entropy
            torch.save(model.state_dict(), best_model_path)
            print(f"   -> ¡Nuevo modelo guardado! (Menor entropía: {best_target_entropy:.4f})")
            
    # --- Finalización y Exportación ---
    df_history = pd.DataFrame(history)
    csv_path = f"./logs/history_{method}_{source_name}_to_{target_name}_seed{seed}.csv"
    df_history.to_csv(csv_path, index=False)
    
    # Cargar los pesos del mejor modelo (el de menor entropía) antes de devolverlo para inferencia
    model.load_state_dict(torch.load(best_model_path))
    return model
