#!/bin/bash

# HSDA - Script de Inferência Rápida
# Este script executa o teste e visualização do modelo HSDA

echo "=== HSDA Inference Script ==="
echo "Iniciando teste e visualização do modelo..."

# Verificar se o container existe
if ! sudo docker ps -a | grep -q hsda_container; then
    echo "Container não encontrado. Criando novo container..."
    sudo docker run -it -d --gpus all --name hsda_container \
        -v $(pwd):/mmdetection3d \
        open-mmlab/mmdetection3d:latest
else
    echo "Container encontrado. Verificando status..."
    if ! sudo docker ps | grep -q hsda_container; then
        echo "Iniciando container existente..."
        sudo docker start hsda_container
    fi
fi

echo "Executando teste do modelo..."
sudo docker exec -i hsda_container bash -c "
cd /mmdetection3d && \
python tools/test.py \
  configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
  checkpoints/epoch_20.pth \
  --eval=bbox \
  --format-only \
  --eval-options jsonfile_prefix=./results_vis
"

if [ $? -eq 0 ]; then
    echo "Teste concluído com sucesso!"
    echo "Gerando visualizações..."
    
    sudo docker exec -i hsda_container bash -c "
    cd /mmdetection3d && \
    python tools/analysis_tools/vis.py \
      results_vis.json \
      --version=val \
      --format=image \
      --vis-thred=0.1 \
      --root_path=data/nuscenes \
      --save_path=./visualizations_all
    "
    
    if [ $? -eq 0 ]; then
        echo "Visualizações geradas com sucesso!"
        echo "Copiando resultados..."
        
        # Criar diretório de resultados se não existir
        mkdir -p ./results
        
        # Copiar resultados
        sudo docker cp hsda_container:/mmdetection3d/visualizations_all ./results/ 2>/dev/null || echo "Visualizações já existem localmente"
        sudo docker cp hsda_container:/mmdetection3d/results_vis.json ./results/ 2>/dev/null || echo "results_vis.json já existe localmente"
        
        echo "=== PROCESSO CONCLUÍDO ==="
        echo "Resultados disponíveis em:"
        echo "  - ./results/visualizations_all/ (81 visualizações)"
        echo "  - ./results/results_vis.json (dados de detecção)"
        
        # Mostrar estatísticas
        if [ -d "./results/visualizations_all" ]; then
            NUM_IMAGES=$(find ./results/visualizations_all -name "*.jpg" | wc -l)
            SIZE=$(du -sh ./results/visualizations_all | cut -f1)
            echo "  - Total de imagens: $NUM_IMAGES"
            echo "  - Tamanho: $SIZE"
        fi
        
    else
        echo "Erro na geração de visualizações!"
        exit 1
    fi
else
    echo "Erro no teste do modelo!"
    exit 1
fi
