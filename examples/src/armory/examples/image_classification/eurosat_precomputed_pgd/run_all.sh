#!/usr/bin/env bash

THIS_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR

MODELS="convnext swin vit"

for model in $MODELS; do
    echo "================================"
    echo "Creating adversarial dataset for $model..."
    echo "================================"
    python create_adversarial_eurosat.py $model

    echo "================================"
    echo "Performing inline attack evaluation for $model..."
    echo "================================"
    python run_with_attack.py $model
done

for eval_model in $MODELS; do
    for attack_model in $MODELS; do
        echo "================================"
        echo "Performing precomputed $attack_model attack evaluation for $eval_model..."
        echo "================================"
        python run_against_pregenerated.py $eval_model eurosat_with_${attack_model}_pgd
    done
done

echo "Done!"
