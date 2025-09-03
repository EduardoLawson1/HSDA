#!/bin/bash
# Script para testar HSDA sem dependências de mapa

echo "🚀 HSDA Test Script (No Maps Required)"
echo "====================================="

CONFIG="configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py"
CHECKPOINT="checkpoints/epoch_20.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_simple_test_$TIMESTAMP"

mkdir -p $OUTPUT_DIR

echo "📊 Executando teste simples (apenas detecção)..."

# Comando simplificado sem avaliação completa
sudo docker exec -it hsda_container bash -c "
cd /mmdetection3d && 
python tools/test.py \
    /mmdetection3d/$CONFIG \
    /mmdetection3d/$CHECKPOINT \
    --out /mmdetection3d/$OUTPUT_DIR/predictions.pkl \
    --show-dir /mmdetection3d/$OUTPUT_DIR/visualizations
" 2>&1 | tee $OUTPUT_DIR/test_log.txt

echo ""
echo "✅ Teste concluído!"
echo "📁 Resultados em: $OUTPUT_DIR/"
echo "🖼️ Visualizações em: $OUTPUT_DIR/visualizations/"

# Verificar quantas predições foram geradas
if [ -f "$OUTPUT_DIR/predictions.pkl" ]; then
    echo "📦 Arquivo de predições gerado com sucesso"
else
    echo "⚠️ Arquivo de predições não encontrado"
fi

echo ""
echo "📋 Para ver métricas detalhadas, use:"
echo "python -c \"import pickle; import mmcv; data=mmcv.load('$OUTPUT_DIR/predictions.pkl'); print(f'Predições geradas: {len(data)}')\"" 
