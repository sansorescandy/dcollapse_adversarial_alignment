import argparse
import random
import numpy as np
import torch
import torch.nn as nn

# Importar nuestros módulos
from data_loader import get_dataloaders
from models import DANNModel, CDANModel
from trainer import train_dann
from flattening_metrics import diagnose_feature_flattening

def set_seed(seed):
    """
    Fija todas las semillas posibles para garantizar reproducibilidad exacta.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features_and_evaluate(model, dataloader, device):
    """
    Extrae los vectores de 256 dimensiones (features) y calcula la exactitud de clasificación.
    """
    model.eval()
    all_features = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Pasamos alpha=0 porque en inferencia no nos importa el GRL
            class_logits, _, features = model(x, alpha=0.0)
            
            # Guardar para el cálculo del FFI
            all_features.append(features.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
            # Calcular Accuracy
            _, predicted = torch.max(class_logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    accuracy = 100.0 * correct / total
    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    
    return features_np, labels_np, accuracy

def main():
    parser = argparse.ArgumentParser(description="Capacity-Aware UDA Experiments")
    parser.add_argument('--method', type=str, required=True, choices=['Source', 'DANN', 'CDAN'], help="Método de adaptación")
    parser.add_argument('--source', type=str, required=True, help="Dominio fuente (ej. amazon)")
    parser.add_argument('--target', type=str, required=True, help="Dominio destino (ej. webcam)")
    parser.add_argument('--data_dir', type=str, default='./data/office31', help="Ruta al dataset")
    parser.add_argument('--seed', type=int, default=42, help="Semilla aleatoria")
    parser.add_argument('--epochs', type=int, default=50, help="Número de épocas")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamaño del lote")
    args = parser.parse_args()

    # 1. Preparar el entorno
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando Experimento: {args.method} | {args.source} -> {args.target} | Seed: {args.seed} ---")
    print(f"Dispositivo: {device}")

    # 2. Cargar Datos
    # El dominio fuente nos da train y val (usamos val_source para extraer métricas baseline si se desea)
    train_source, val_source = get_dataloaders(args.data_dir, args.source, batch_size=args.batch_size, is_source=True)
    # El dominio destino nos da un loader para adaptación y validación final
    target_loader = get_dataloaders(args.data_dir, args.target, batch_size=args.batch_size, is_source=False)

    # 3. Instanciar Modelo
    if args.method == 'CDAN':
        model = CDANModel(num_classes=31).to(device)
    else:
        # Usamos DANNModel tanto para DANN como para Source-Only.
        # (Para Source-Only, simplemente ignoraremos la pérdida de dominio en un loop especializado, 
        # o podemos usar train_dann con alpha=0 forzado. Por simplicidad del código modular, 
        # asumiremos que train_dann maneja la lógica DANN/CDAN).
        model = DANNModel(num_classes=31).to(device)

    # 4. Entrenamiento
    if args.method in ['DANN', 'CDAN']:
        # Utilizamos nuestro trainer optimizado con Target Entropy
        model = train_dann(model, train_source, target_loader, num_epochs=args.epochs, device=device)
    else:
        print("Entrenando Source-Only... (Implementar loop estándar de clasificación aquí)")
        # Aquí iría un bucle simple de CrossEntropyLoss clásico sin la rama del discriminador.

    # 5. Evaluación Final y Extracción de Tensores
    print("Extrayendo representaciones para diagnóstico de Feature Flattening...")
    
    # Extraemos características del dominio fuente (usamos val_source) y del dominio destino
    Z_s, y_s, acc_source = extract_features_and_evaluate(model, val_source, device)
    Z_t, y_t, acc_target = extract_features_and_evaluate(model, target_loader, device)

    print(f"Precisión Final en Destino ({args.target}): {acc_target:.2f}%")

    # 6. Calcular Métricas de Feature Flattening
    # Nota: Para usar `diagnose_feature_flattening` estrictamente, necesitaríamos las features
    # del modelo Baseline (Source-Only). En este script evaluamos el FFI del modelo actual.
    from flattening_metrics import ffi
    
    d = Z_s.shape[1] # Debería ser 256
    metricas_fuente = ffi(Z_s, y_s, d)
    # Para el dominio destino usamos las pseudo-etiquetas generadas por el modelo (y_t) para medir separabilidad
    metricas_destino = ffi(Z_t, y_t, d) 

    print("\n--- Resultados Matemáticos del Espacio de Características ---")
    print(f"Dominio Fuente  -> erank: {metricas_fuente['erank']:.2f}, DR: {metricas_fuente['DR']:.2f}, FFI: {metricas_fuente['FFI']:.4f}")
    print(f"Dominio Destino -> erank: {metricas_destino['erank']:.2f}, DR: {metricas_destino['DR']:.2f}, FFI: {metricas_destino['FFI']:.4f}")

    # --- NUEVO: Guardar el modelo físicamente en el disco ---
    import os
    os.makedirs("./checkpoints", exist_ok=True)
    nombre_archivo = f"./checkpoints/best_model_{args.method}_{args.source}_to_{args.target}_seed{args.seed}.pth"
    torch.save(model.state_dict(), nombre_archivo)
    print(f"\nModelo guardado en: {nombre_archivo}")

if __name__ == '__main__':
    main()
