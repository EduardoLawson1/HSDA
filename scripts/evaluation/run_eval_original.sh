#!/bin/bash

# Script para executar avaliação com métricas mIoU reais
echo "🧪 Executando avaliação HSDA com métricas mIoU originais..."

cd /mmdetection3d

# Tentar execução simples sem mapas (apenas detecção)
echo "1️⃣ Tentando avaliação apenas com bbox..."
python tools/test.py \
    configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
    checkpoints/epoch_20.pth \
    --eval bbox \
    --out results_bbox_metrics.pkl \
    2>&1 | tee evaluation_bbox.log

echo "2️⃣ Verificando arquivos gerados..."
ls -la *.pkl *.log 2>/dev/null || echo "Nenhum arquivo de resultado encontrado"

echo "✅ Avaliação bbox concluída!"
