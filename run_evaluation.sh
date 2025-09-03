#!/bin/bash
# Script para executar teste do HSDA com métricas

echo "🚀 HSDA Model Evaluation Script"
echo "================================"

CONFIG="configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py"
CHECKPOINT="checkpoints/epoch_20.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_evaluation_$TIMESTAMP"

# Criar diretório de resultados
mkdir -p $OUTPUT_DIR

echo "📊 Configuração:"
echo "  - Config: $CONFIG"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Output: $OUTPUT_DIR"
echo ""

# Verificar se arquivos existem
if [ ! -f "$CONFIG" ]; then
    echo "❌ Erro: Arquivo de configuração não encontrado: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Erro: Checkpoint não encontrado: $CHECKPOINT"
    exit 1
fi

echo "🔍 Executando avaliação do modelo..."

# Executar dentro do container Docker
sudo docker exec -it hsda_container bash -c "
cd /mmdetection3d && 
python tools/test.py \
    /mmdetection3d/$CONFIG \
    /mmdetection3d/$CHECKPOINT \
    --eval=bbox \
    --out /mmdetection3d/$OUTPUT_DIR/predictions.pkl
" 2>&1 | tee $OUTPUT_DIR/evaluation_log.txt

# Verificar se a execução foi bem-sucedida
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Avaliação concluída com sucesso!"
    echo "📁 Resultados salvos em: $OUTPUT_DIR/"
    echo "📊 Log completo: $OUTPUT_DIR/evaluation_log.txt"
    echo ""
    echo "📈 Resumo das métricas:"
    echo "======================"
    # Extrair métricas do log
    grep -E "(mAP|NDS|AP)" $OUTPUT_DIR/evaluation_log.txt || echo "Métricas detalhadas no arquivo de log"
else
    echo ""
    echo "❌ Erro durante a avaliação!"
    echo "📋 Verifique o log em: $OUTPUT_DIR/evaluation_log.txt"
    echo ""
    echo "🔧 Possíveis soluções:"
    echo "  1. Verificar se o container está rodando: sudo docker ps"
    echo "  2. Verificar se os dados estão montados corretamente"
    echo "  3. Verificar se há problemas de memória"
fi

echo ""
echo "🏁 Script finalizado."
