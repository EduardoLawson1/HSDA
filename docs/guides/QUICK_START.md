# HSDA - Guia Rápido de Execução

## Pré-requisitos

- Docker instalado
- NVIDIA Docker runtime configurado
- Dataset nuScenes completo em `data/nuscenes/`
- Checkpoint do modelo em `checkpoints/epoch_20.pth`
- Arquivos de mapa nuScenes em `data/nuscenes/maps/expansion/`

## Estrutura Necessária

```
HSDA/
├── data/nuscenes/
│   ├── maps/expansion/
│   │   ├── singapore-onenorth.json
│   │   ├── singapore-hollandvillage.json
│   │   ├── singapore-queenstown.json
│   │   └── boston-seaport.json
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-trainval/
│   └── nuscenes_infos_val.pkl
├── checkpoints/
│   └── epoch_20.pth
└── configs/bevdet_hsda/
    └── bevdet-multi-map-aug-seg-only-6class-hsda.py
```

## Execução Automatizada (Recomendado)

### Script de Inferência Completa
```bash
# Executa teste + visualização + cópia de resultados automaticamente
./run_inference.sh
```

Este script automatiza todo o processo:
1. Verifica/cria/inicia o container Docker
2. Executa o teste do modelo
3. Gera todas as visualizações
4. Copia resultados para `./results/`
5. Exibe estatísticas finais

## Comandos de Execução

### 1. Iniciar Container Docker
```bash
sudo docker run -it --gpus all --name hsda_container \
  -v /home/pdi/Documentos/autonomi/HSDA:/mmdetection3d \
  open-mmlab/mmdetection3d:latest
```

### 2. Acesso ao Container
```bash
sudo docker exec -it hsda_container bash
```

### 3. Teste do Modelo (Dentro do Container)
```bash
python tools/test.py \
  configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
  checkpoints/epoch_20.pth \
  --eval=bbox \
  --format-only \
  --eval-options jsonfile_prefix=./results_vis
```

### 4. Visualizações Completas (Dentro do Container)
```bash
python tools/analysis_tools/vis.py \
  results_vis.json \
  --version=val \
  --format=image \
  --vis-thred=0.1 \
  --root_path=data/nuscenes \
  --save_path=./visualizations_all
```

### 5. Copiar Resultados para Host
```bash
# Sair do container
exit

# Copiar visualizações
sudo docker cp hsda_container:/mmdetection3d/visualizations_all ./results/

# Copiar arquivo de resultados
sudo docker cp hsda_container:/mmdetection3d/results_vis.json ./results/
```

## Comandos Alternativos (Uma Linha)

### Teste Completo
```bash
sudo docker exec -it hsda_container bash -c "
cd /mmdetection3d && \
python tools/test.py \
  configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
  checkpoints/epoch_20.pth \
  --eval=bbox \
  --format-only \
  --eval-options jsonfile_prefix=./results_vis
"
```

### Visualização Completa
```bash
sudo docker exec -it hsda_container bash -c "
cd /mmdetection3d && \
python tools/analysis_tools/vis.py \
  results_vis.json \
  --version=val \
  --format=image \
  --vis-thred=0.1 \
  --root_path=data/nuscenes \
  --save_path=./visualizations_all
"
```

## Arquivos de Configuração Testados

### Principal (Recomendado)
- `bevdet-multi-map-aug-seg-only-6class-hsda.py` - Configuração completa funcional

### Alternativas
- `bevdet-hsda-no-maps.py` - Versão sem mapas
- `bevdet-multi-no-tta.py` - Versão sem test-time augmentation

## Resultados Esperados

### Métricas de Performance
- **mAP Overall**: ~0.2833
- **mATE (Translation Error)**: 0.7733
- **mASE (Scale Error)**: 0.2798
- **Classes principais**: car (0.5156), barrier (0.5005), traffic_cone (0.4963)

### Visualizações Geradas
- **Quantidade**: 81 frames do dataset de validação
- **Formato**: JPG multi-view (6 câmeras + BEV)
- **Tamanho**: ~29MB total
- **Localização**: `./results/visualizations_all/`

### Arquivos de Saída
- `results_vis.json` - Dados de detecção formatados para visualização
- `visualizations_all/` - Diretório com todas as imagens geradas

## Problema Conhecido

⚠️ **Detecção de Objetos**: O modelo atual (epoch_20.pth) apresenta limitações:
- Caixas de detecção concentradas em uma região
- Scores idênticos (0.502) para todas as detecções
- Necessário re-treinamento para detecção funcional

✅ **Segmentação BEV**: Funciona corretamente

## Troubleshooting

### Erro: singapore-onenorth.json não encontrado
```bash
# Verificar estrutura do dataset
find data/nuscenes/maps -name "*.json"
```

### Container não inicia
```bash
# Verificar containers existentes
sudo docker ps -a

# Remover container antigo se necessário
sudo docker rm hsda_container
```

### Memória insuficiente
```bash
# Limpar containers antigos
sudo docker system prune

# Verificar espaço em disco
df -h
```

## Status do Projeto

- ✅ **Framework**: Configurado e funcional
- ✅ **Dataset**: Organizado e verificado
- ✅ **Visualização**: Pipeline completo implementado
- ⚠️ **Modelo**: Necessita re-treinamento para detecção adequada
- ✅ **Documentação**: Completa e atualizada
