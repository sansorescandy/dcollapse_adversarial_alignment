import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_target_entropy(model, dataloader, device):
    """Calculates Shannon entropy on the target domain (without labels)."""
    model.eval()
    entropies = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            # Extract class logits, ignore domain logits and features
            class_logits, _, _ = model(x, alpha=0.0)
            
            # Entropy calculation
            probs = F.softmax(class_logits, dim=1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            entropies.append(batch_entropy.item())
            
    return sum(entropies) / len(entropies)

def evaluate_source_domain(model, dataloader, device):
    """Evaluates classification loss and accuracy on a given domain."""
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

def train_adversarial(model, train_source, val_source, target_loader, num_epochs, device, method, source_name, target_name, seed):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCELoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    history = {
        'epoch': [], 'train_loss_cls': [], 'train_loss_domain': [], 'train_acc': [],
        'val_source_loss': [], 'val_source_acc': [], 
        'val_target_entropy': [],
        'val_target_acc': [] # Silent Oracle tracking for ablation
    }
    
    best_target_entropy = float('inf')
    
    # Official checkpoint path (removed 'ablation_')
    best_model_path = f"./checkpoints/best_model_{method}_{source_name}_to_{target_name}_seed{seed}.pth"
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss_cls = 0.0
        epoch_loss_domain = 0.0
        correct_cls = 0 # Accuracy counter
        total_samples = 0 # Sample counter
        
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + float(np.exp(-10. * p))) - 1
        
        len_dataloader = min(len(train_source), len(target_loader))
        source_iter = iter(train_source)
        target_iter = iter(target_loader)
        
        for i in range(len_dataloader):
            s_img, s_label = next(source_iter)
            s_img, s_label = s_img.to(device), s_label.to(device)
            
            t_img, _ = next(target_iter)
            t_img = t_img.to(device)
            
            optimizer.zero_grad()
            
            # --- Forward Pass (Source) ---
            class_logits, domain_logits_s, _ = model(s_img, alpha=alpha)
            loss_cls = criterion_cls(class_logits, s_label)
            
            # Calculate training accuracy
            _, predicted = torch.max(class_logits.data, 1)
            correct_cls += (predicted == s_label).sum().item()
            
            domain_label_s = torch.zeros(domain_logits_s.size(0), 1).to(device)
            loss_domain_s = criterion_domain(domain_logits_s, domain_label_s)
            
            # --- Forward Pass (Target) ---
            _, domain_logits_t, _ = model(t_img, alpha=alpha)
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
        train_acc = 100.0 * correct_cls / total_samples 
        
        # --- Validation Phase ---
        val_src_loss, val_src_acc = evaluate_source_domain(model, val_source, device)
        val_tgt_entropy = calculate_target_entropy(model, target_loader, device)
        
        # Calculate target accuracy (silent oracle tracking)
        _, val_tgt_acc = evaluate_source_domain(model, target_loader, device)

        # Record metrics
        history['epoch'].append(epoch)
        history['train_loss_cls'].append(avg_loss_cls)
        history['train_loss_domain'].append(avg_loss_domain)
        history['train_acc'].append(train_acc) 
        history['val_source_loss'].append(val_src_loss)
        history['val_source_acc'].append(val_src_acc)
        history['val_target_entropy'].append(val_tgt_entropy)
        history['val_target_acc'].append(val_tgt_acc)

        print(f"Epoch [{epoch}/{num_epochs}] | Cls Loss: {avg_loss_cls:.4f} | Train Acc: {train_acc:.2f}% | Src Acc: {val_src_acc:.2f}% | Tgt Entropy: {val_tgt_entropy:.4f} | Tgt Acc: {val_tgt_acc:.4f}")
        
        if val_tgt_entropy < best_target_entropy:
            best_target_entropy = val_tgt_entropy
            # Model selection remains strictly unsupervised (based on entropy)
            torch.save(model.state_dict(), best_model_path)
            
    df_history = pd.DataFrame(history)
    # Log path
    csv_path = f"./logs/history_{method}_{source_name}_to_{target_name}_seed{seed}.csv"
    df_history.to_csv(csv_path, index=False)
    
    model.load_state_dict(torch.load(best_model_path))
    return model
