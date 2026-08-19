#!/bin/bash

# Asegúrate de dar permisos de ejecución a este archivo: chmod +x run_experiments.sh

DATA_DIR="./data/office31"
LOG_DIR="./logs"
CHECKPOINT_DIR="./checkpoints"
EPOCHS=50
# Expandimos a 10 semillas para poder estadístico
SEEDS=(42 10 99 123 777 11 22 33 44 55)
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

echo "Starting UDA Experiment Battery..."

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
            echo "Running: Method=$method | Task=$source_dom->$target_dom | Seed=$seed"
            echo "Progress is silently logged to: $LOG_FILE"
            echo "================================================================="
            
            # Ejecutar Python y redirigir salida estándar y de errores al log
            python main.py \
                --method $method \
                --source $source_dom \
                --target $target_dom \
                --data_dir $DATA_DIR \
                --seed $seed \
                --epochs $EPOCHS 2>&1 | tee "$LOG_FILE"
                
        done
    done
done

echo "All experiments have finished successfully."