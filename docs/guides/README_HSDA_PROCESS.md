# HSDA - Documentação├── results/                   # Resultados finais organizados
│   ├── visualizations/       # Visualizações (5 amostras)
│   ├── visualizations_all/   # Todas as 81 visualizações
│   └── results_vis.json      # Resultados formatados para visualização
├── run_inference.sh          # Script automatizado de inferência
├── README_HSDA_PROCESS.md    # Documentação completa do processo
└── QUICK_START.md           # Guia rápido de execução atualizadoeta do Processo

Este documento detalha todo o processo de configuração, teste e execução do modelo HSDA (Hierarchical Scene Decomposition and Adaptation) para detecção 3D em condução autônoma.

## Visão Geral

O HSDA é um modelo baseado em BEVDet para detecção de objetos 3D em cenários de condução autônoma, utilizando múltiplas câmeras e dados do dataset nuScenes.

## Estrutura do Projeto (Limpa)

```
HSDA/
├── configs/                    # Configurações do modelo
│   └── bevdet_hsda/           # Configurações específicas do HSDA
├── data/                      # Dataset nuScenes
│   └── nuscenes/             # Dados organizados do nuScenes
├── mmdet3d/                   # Framework base MMDetection3D
├── mmdet3d_plugin/           # Plugins específicos do HSDA
├── tools/                     # Ferramentas utilitárias
├── checkpoints/              # Modelos treinados
├── work_dirs/                # Diretórios de trabalho
├── results/                  # Resultados finais organizados
│   ├── visualizations/       # Visualizações (5 amostras)
│   ├── visualizations_all/   # Todas as 81 visualizações
│   └── results_vis.json      # Resultados formatados para visualização
├── README_HSDA_PROCESS.md    # Documentação completa do processo
└── QUICK_START.md           # Guia rápido de execução
```

## Histórico de Problemas e Soluções

### 1. Configuração Inicial do Ambiente

**Problema**: Incompatibilidades de dependências e versões do Python.

**Solução**: 
- Utilizamos Docker com imagem `open-mmlab/mmdetection3d:latest`
- Configuração específica do ambiente conda dentro do container

### 2. Problemas com Dataset nuScenes

**Problema**: Arquivos de mapa não encontrados (`singapore-onenorth.json`).

**Solução**:
- Download e organização correta do dataset nuScenes
- Estrutura de diretórios: `data/nuscenes/maps/expansion/`

**Comando para verificar estrutura**:
```bash
find data/nuscenes -name "*.json" | head -10
```

### 3. Configuração do MMCV

**Problema**: Versões incompatíveis do MMCV.

**Solução**:
```bash
pip uninstall mmcv-full mmcv
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
```

### 4. Problemas de Importação

**Problema**: Módulos `mmdet3d_plugin` não encontrados.

**Solução**:
- Adição do caminho ao PYTHONPATH
- Instalação em modo desenvolvimento:
```bash
pip install -e .
```

### 5. Configuração CUDA

**Problema**: CUDA não detectado corretamente.

**Solução**:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

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

### 3. Teste do Modelo

```bash
python tools/test.py \
  configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
  checkpoints/epoch_20.pth \
  --eval=bbox \
  --out results_nusc.pkl
```

### 4. Avaliação com Métricas

```bash
python tools/test.py \
  configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
  checkpoints/epoch_20.pth \
  --eval=bbox \
  --format-only \
  --eval-options jsonfile_prefix=./results_formatted
```

### 5. Geração de Visualizações

#### Visualizações Limitadas (5 amostras)
```bash
python tools/analysis_tools/vis.py \
  results_vis.json \
  --version=val \
  --format=image \
  --vis-frames=5 \
  --vis-thred=0.1
```

#### Todas as Visualizações (81 amostras)
```bash
python tools/analysis_tools/vis.py \
  results_vis.json \
  --version=val \
  --format=image \
  --vis-thred=0.1 \
  --root_path=data/nuscenes \
  --save_path=./visualizations_all
```

## Resultados Obtidos

### Métricas de Performance

O modelo HSDA apresentou os seguintes resultados no dataset de validação nuScenes:

**mAP (mean Average Precision):**
- Overall mAP: 0.2833
- mATE (Translation Error): 0.7733
- mASE (Scale Error): 0.2798
- mAOE (Orientation Error): 0.5485
- mAVE (Velocity Error): 0.8950
- mAAE (Attribute Error): 0.2001

**mAP por Classe:**
- car: 0.5156
- truck: 0.2435
- bus: 0.3267
- trailer: 0.1165
- construction_vehicle: 0.0646
- pedestrian: 0.3661
- motorcycle: 0.2269
- bicycle: 0.1434
- traffic_cone: 0.4963
- barrier: 0.5005

### Visualizações Geradas

- **5 amostras**: Pasta `visualizations/`
- **81 amostras completas**: Pasta `visualizations_all/`
- **Formato**: Imagens JPG com múltiplas vistas (6 câmeras + BEV)
- **Tamanho total**: ~29MB

## Arquivos de Configuração Importantes

### 1. bevdet-multi-map-aug-seg-only-6class-hsda.py

Configuração principal do modelo HSDA com:
- Arquitetura BEVDet
- Configurações de dataset nuScenes
- Parâmetros de treinamento e teste
- Augmentações específicas

### 2. Dataset Configuration

Localização dos dados:
- `data_root = 'data/nuscenes/'`
- Arquivos de anotação: `nuscenes_infos_val.pkl`
- Mapas: `data/nuscenes/maps/expansion/`

## Comandos de Limpeza e Manutenção

### Copiar Resultados do Container

```bash
# Copiar visualizações
sudo docker cp hsda_container:/mmdetection3d/visualizations_all ./visualizations_all

# Copiar resultados
sudo docker cp hsda_container:/mmdetection3d/results_vis.json ./results_vis.json
```

### Verificar Status do Container

```bash
sudo docker ps -a
sudo docker exec -it hsda_container bash -c "ls -la"
```

## Próximos Passos

1. **Limpeza do Projeto**: Remover arquivos temporários e desnecessários
2. **Organização**: Manter apenas arquivos essenciais do projeto original
3. **Documentação**: Criar guia simplificado para execução
4. **Otimização**: Estudar possibilidades de melhoria do modelo

## Troubleshooting

### Problema: Container não inicia
```bash
sudo docker restart hsda_container
```

### Problema: Memória insuficiente
- Verificar espaço em disco
- Limpar containers antigos: `sudo docker system prune`

### Problema: CUDA não detectado
```bash
nvidia-smi  # Verificar GPUs disponíveis
export CUDA_VISIBLE_DEVICES=0
```

### Problema: Dependências não encontradas
```bash
pip install -r requirements.txt
pip install -e .
```

## Estrutura de Dados Necessária

```
data/nuscenes/
├── maps/
│   └── expansion/
│       ├── singapore-onenorth.json
│       ├── singapore-hollandvillage.json
│       ├── singapore-queenstown.json
│       └── boston-seaport.json
├── samples/
├── sweeps/
├── v1.0-trainval/
└── nuscenes_infos_val.pkl
```

## Problema Crítico Identificado: Caixas de Detecção

### **Descrição do Problema**

Durante a análise das visualizações, foi identificado um problema crítico com as caixas de detecção 3D:

**Sintomas Observados**:
- Todas as caixas de detecção aparecem concentradas em uma única região das imagens
- Caixas visíveis apenas na câmera frontal (CAM_FRONT)
- Distribuição espacial irreal dos objetos detectados
- Todas as detecções têm score idêntico (0.502)

### **Análise Técnica**

**1. Coordenadas das Predições**:
- Modelo produz coordenadas em sistema **global** (GPS-like): `[645.25, 1626.37, -1.22]`
- Ground truth está em coordenadas **lidar** (relativas): `[-7.96, 36.90, 0.21]`
- Offset global do veículo: `[601.01, 1646.99, 1.82]`

**2. Distribuição Espacial**:
```
Total de detecções: 1053 (em 81 frames)
Score mínimo: 0.502
Score máximo: 0.502
Variação espacial: Extremamente baixa (~20m de range)
```

**3. Transformação de Coordenadas**:
- Transformação global→lidar funciona corretamente
- Problema não está na visualização ou transformação
- Todas as predições ficam na região ~48m à frente do veículo

### **Causa Raiz do Problema**

O modelo HSDA (epoch_20.pth) **não está adequadamente treinado** para detecção de objetos individuais:

1. **Scores idênticos**: Indica convergência inadequada ou overfitting
2. **Posições concentradas**: Modelo não aprendeu variação espacial realística
3. **Ausência de diversidade**: Não detecta objetos em diferentes regiões da cena

### **Impacto no Projeto**

- ❌ **Detecção de objetos individuais**: Não funcional
- ❌ **Localização espacial**: Irreal
- ✅ **Segmentação BEV**: Funciona corretamente
- ✅ **Pipeline de visualização**: Implementado e funcional

### **Soluções Recomendadas**

1. **Re-treinar o modelo**:
   ```bash
   python tools/train.py configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
     --work-dir work_dirs/bevdet_hsda_retrain --epochs 50
   ```

2. **Usar modelo pré-treinado**:
   - Buscar checkpoint oficial do BEVDet
   - Verificar modelo base antes da adaptação HSDA

3. **Ajustar configurações**:
   - Aumentar `score_threshold` no test_cfg
   - Modificar parâmetros de NMS
   - Verificar configuração do dataset

### **Status Atual**

- **Visualizações geradas**: 81 frames processados
- **Problema documentado**: Análise técnica completa
- **Projeto limpo**: Arquivos temporários removidos (02/09/2025)
- **Próximo passo**: Re-treinamento necessário para detecção funcional

## Limpeza do Projeto (02/09/2025)

### **Arquivos Removidos na Limpeza**
- **Cache Python**: Todos os `__pycache__/` e `*.pyc` removidos
- **Configurações temporárias**: 5 arquivos de teste em `configs/bevdet_hsda/`
- **Scripts de debug**: 4 arquivos `vis_*_*.py` em `tools/analysis_tools/`
- **Build artifacts**: Diretório `build/` e `mmdet3d.egg-info/`
- **Arquivos temporários**: Logs, cache e arquivos de desenvolvimento

### **Arquivos Essenciais Preservados**
```
HSDA/                           # 9.0GB total
├── checkpoints/                # 1.1GB - Modelo treinado
│   └── epoch_20.pth           # Checkpoint principal
├── configs/                    # 332KB - Configurações essenciais
│   └── bevdet_hsda/           # 6 arquivos de configuração válidos
├── data/                      # 7.8GB - Dataset nuScenes
├── mmdet3d/                   # 105MB - Framework core
├── mmdet3d_plugin/            # 300KB - Plugins HSDA
├── results/                   # 30MB - Resultados finais
│   ├── results_vis.json       # Dados de detecção
│   └── visualizations_all/    # 81 visualizações completas
└── tools/                     # 420KB - Ferramentas principais
    ├── analysis_tools/        # 4 arquivos essenciais
    │   ├── vis.py            # Script usado na inferência bem-sucedida
    │   ├── analyze_logs.py   # Análise de logs
    │   ├── benchmark.py      # Benchmark
    │   └── get_flops.py      # Calculadora de FLOPs
    ├── data_converter/       # Conversores de dados
    ├── misc/                 # Utilitários diversos
    └── model_converters/     # Conversores de modelo
```

### **Configurações Mantidas**
- `bevdet-multi-map-aug-seg-only-6class-hsda.py` (principal)
- `bevdet-multi-map-aug-seg-only-6class-rgc-hsda.py`
- `bevdet-hsda-no-maps.py`
- `bevdet-multi-no-tta.py`
- Outras configurações base válidas

## Comandos Úteis

```bash
# Verificar estrutura do dataset
find data/nuscenes -type f -name "*.json" | wc -l

# Monitorar uso de GPU
watch -n 1 nvidia-smi

# Verificar logs do container
sudo docker logs hsda_container

# Backup dos resultados
tar -czf hsda_results_$(date +%Y%m%d).tar.gz visualizations_all/ results_vis.json

# Analisar qualidade das detecções
python -c "
import json
res = json.load(open('results/results_vis.json', 'r'))
scores = [p['detection_score'] for pred_res in res['results'].values() for p in pred_res]
print(f'Scores: min={min(scores):.3f}, max={max(scores):.3f}, unique={len(set(scores))}')
"
```

---

**Data de criação**: 28 de agosto de 2025  
**Última atualização**: 29 de agosto de 2025  
**Status**: Projeto funcional com problema crítico identificado na detecção de objetos
