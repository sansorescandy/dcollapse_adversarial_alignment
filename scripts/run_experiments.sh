#!/bin/bash

# Asegúrate de dar permisos de ejecución a este archivo: chmod +x run_experiments.sh

DATA_DIR="./data/office31"
LOG_DIR="./logs"
CHECKPOINT_DIR="./checkpoints"
EPOCHS=50
SEEDS=(42 10 99 123 777)
METHODS=("Source" "DANN" "CDAN")
TASKS=(
    "amazon webcam"
    "amazon dslr"
    "webcam amazon"
    "webcam dslr"
    "dslr amazon"
    "dslr webcam"
)

# 1. Crear las carpetas si no existen
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

echo "Iniciando Batería de Experimentos UDA..."

# Bucle triple: Método -> Tarea -> Semilla
for method in "${METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        # Separar la tarea en fuente y destino
        set -- $task
        source_dom=$1
        target_dom=$2
        
        for seed in "${SEEDS[@]}"; do
            # Generar un nombre único para el archivo log
            LOG_FILE="${LOG_DIR}/${method}_${source_dom}_to_${target_dom}_seed${seed}.log"
            
            echo "================================================================="
            echo "Ejecutando: Método=$method | Tarea=$source_dom->$target_dom | Seed=$seed"
            echo "El progreso se está guardando silenciosamente en: $LOG_FILE"
            echo "================================================================="
            
            # Ejecutar Python y redirigir salida estándar y de errores al log
            python main.py \
                --method $method \
                --source $source_dom \
                --target $target_dom \
                --data_dir $DATA_DIR \
                --seed $seed \
                --epochs $EPOCHS > "$LOG_FILE" 2>&1
                
        done
    done
done

echo "Todos los experimentos han finalizado exitosamente."
