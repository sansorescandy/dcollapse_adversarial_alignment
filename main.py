import sys
import os
import pandas as pd
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Import our custom modules
from data_loader import get_dataloaders
from models import DANNModel, CDANModel
from trainer import train_adversarial 
from flattening_metrics import diagnose_feature_flattening, ffi

def set_seed(seed):
    """
    Sets all possible random seeds to ensure exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features_and_evaluate(model, dataloader, device):
    """
    Extracts the 256-dimensional feature vectors and calculates classification accuracy.
    """
    model.eval()
    all_features = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Pass alpha=0 because GRL is not needed during inference
            class_logits, _, features = model(x, alpha=0.0)
            
            # Save for FFI calculation
            all_features.append(features.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
            # Calculate Accuracy
            _, predicted = torch.max(class_logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    accuracy = 100.0 * correct / total
    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    
    return features_np, labels_np, accuracy

def evaluate_source_domain(model, dataloader, device):
    """
    Evaluates loss and accuracy on a domain where labels are available.
    """
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

def calculate_target_entropy(model, dataloader, device):
    """
    Calculates Shannon Entropy on the target domain predictions (Unsupervised).
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            class_logits, _, _ = model(x, alpha=0.0)
            
            probs = F.softmax(class_logits, dim=1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            entropies.append(batch_entropy.item())
            
    return sum(entropies) / len(entropies)

def main():
    parser = argparse.ArgumentParser(description="Capacity-Aware UDA Experiments")
    parser.add_argument('--method', type=str, required=True, choices=['Source', 'DANN', 'CDAN'], help="Adaptation method")
    parser.add_argument('--source', type=str, required=True, help="Source domain (e.g., amazon)")
    parser.add_argument('--target', type=str, required=True, help="Target domain (e.g., webcam)")
    parser.add_argument('--data_dir', type=str, default='./data/office31', help="Dataset root directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # --- EXPERIMENT RESUME LOGIC ---
    summary_file = "./logs/resumen_resultados_global.csv"
    
    if os.path.isfile(summary_file):
        try:
            df_summary = pd.read_csv(summary_file)
            
            # Check for an exact match of the current experiment configuration
            already_processed = df_summary[
                (df_summary['Metodo'] == args.method) & 
                (df_summary['Origen'] == args.source) & 
                (df_summary['Destino'] == args.target) & 
                (df_summary['Seed'] == args.seed)
            ]
            
            if not already_processed.empty:
                print(f"⏭️ Skipping: Experiment {args.method} ({args.source} -> {args.target} | Seed {args.seed}) is already completed.")
                sys.exit(0) 
                
        except Exception as e:
            print(f"Warning: Could not verify the summary log. Proceeding with execution. Error: {e}")
    # -------------------------------

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Experiment: {args.method} | {args.source} -> {args.target} | Seed: {args.seed} ---")

    train_source, val_source = get_dataloaders(args.data_dir, args.source, batch_size=args.batch_size, is_source=True)
    target_loader = get_dataloaders(args.data_dir, args.target, batch_size=args.batch_size, is_source=False)

    if args.method == 'CDAN':
        model = CDANModel(num_classes=31).to(device)
    else:
        model = DANNModel(num_classes=31).to(device)

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    best_model_path = f"./checkpoints/best_model_{args.method}_{args.source}_to_{args.target}_seed{args.seed}.pth"

    if args.method in ['DANN', 'CDAN']:
        model = train_adversarial(model, train_source, val_source, target_loader, args.epochs, device, args.method, args.source, args.target, args.seed)
    else:
        print("Training Source-Only with metric tracking...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([
            {'params': model.backbone.parameters(), 'lr': 0.001},
            {'params': model.classifier.parameters(), 'lr': 0.01}
        ], momentum=0.9, weight_decay=5e-4)
        
        history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_source_loss': [], 'val_source_acc': [], 
            'val_target_entropy': [],
            'val_target_acc': [] 
        }
        
        best_target_entropy = float('inf')
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            correct_train = 0
            total_samples = 0
            
            for x, y in train_source:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                class_logits, _, _ = model(x, alpha=0.0)
                loss = criterion(class_logits, y)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
                
                _, predicted = torch.max(class_logits.data, 1)
                correct_train += (predicted == y).sum().item()
                total_samples += x.size(0)
                
            train_loss_avg = epoch_loss / total_samples
            train_acc = 100.0 * correct_train / total_samples
            
            val_src_loss, val_src_acc = evaluate_source_domain(model, val_source, device)
            val_tgt_entropy = calculate_target_entropy(model, target_loader, device)
            
            # Silent Oracle tracking
            _, val_tgt_acc = evaluate_source_domain(model, target_loader, device)
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_source_loss'].append(val_src_loss)
            history['val_source_acc'].append(val_src_acc)
            history['val_target_entropy'].append(val_tgt_entropy)
            history['val_target_acc'].append(val_tgt_acc)
            
            print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | Src Acc: {val_src_acc:.2f}% | Tgt Entropy: {val_tgt_entropy:.4f} | Tgt Acc: {val_tgt_acc:.2f}%")
            
            if val_tgt_entropy < best_target_entropy:
                best_target_entropy = val_tgt_entropy
                torch.save(model.state_dict(), best_model_path)
                
        df_history = pd.DataFrame(history)
        csv_path = f"./logs/history_{args.method}_{args.source}_to_{args.target}_seed{args.seed}.csv"
        df_history.to_csv(csv_path, index=False)
        model.load_state_dict(torch.load(best_model_path))

    # --- Final Evaluation ---
    print("\nExtracting representations for Feature Flattening diagnosis...")
    Z_s, y_s, acc_source = extract_features_and_evaluate(model, val_source, device)
    Z_t, y_t, acc_target = extract_features_and_evaluate(model, target_loader, device)

    print(f"Final Target Accuracy ({args.target}): {acc_target:.2f}%")

    d = Z_s.shape[1] 
    metricas_fuente = ffi(Z_s, y_s, d)
    metricas_destino = ffi(Z_t, y_t, d) 

    # --- Read history and extract Oracle peak ---
    csv_path = f"./logs/history_{args.method}_{args.source}_to_{args.target}_seed{args.seed}.csv"
    try:
        df_hist = pd.read_csv(csv_path)
        oracle_acc = df_hist['val_target_acc'].max()
    except Exception as e:
        print(f"Warning: Could not read history for Oracle tracking. Using acc_target. Error: {e}")
        oracle_acc = acc_target

    print("\n--- Feature Space Mathematical Results ---")
    print(f"Source Domain -> erank: {metricas_fuente['erank']:.2f}, DR: {metricas_fuente['DR']:.2f}, FFI: {metricas_fuente['FFI']:.4f}")
    print(f"Target Domain -> erank: {metricas_destino['erank']:.2f}, DR: {metricas_destino['DR']:.2f}, FFI: {metricas_destino['FFI']:.4f}")
    print(f"Oracle Gap -> UDA: {acc_target:.2f}% | Oracle: {oracle_acc:.2f}%")

    # --- Export to global summary ---
    resumen_file = "./logs/resumen_resultados_global.csv"
    file_exists = os.path.isfile(resumen_file)
    with open(resumen_file, 'a') as f:
        if not file_exists:
            f.write("Metodo,Origen,Destino,Seed,Target_Acc,Oracle_Acc,Src_Erank,Src_DR,Src_FFI,Tgt_Erank,Tgt_DR,Tgt_FFI\n")
        
        f.write(f"{args.method},{args.source},{args.target},{args.seed},{acc_target:.2f},{oracle_acc:.2f},{metricas_fuente['erank']:.2f},{metricas_fuente['DR']:.2f},{metricas_fuente['FFI']:.4f},{metricas_destino['erank']:.2f},{metricas_destino['DR']:.2f},{metricas_destino['FFI']:.4f}\n")
    print(f"Final metrics successfully appended to: {resumen_file}")

if __name__ == '__main__':
    main()
