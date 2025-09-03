# 🔍 Inference Results

Esta pasta contém os resultados brutos de inferência do modelo HSDA.

## 📁 Arquivos Disponíveis

### `results_vis.json` (487KB)
- **Tipo**: Resultados de detecção de objetos 3D
- **Conteúdo**: 81 amostras do dataset nuScenes
- **Formato**: JSON estruturado
- **Classes detectadas**: car, truck, bus, barrier, motorcycle, bicycle, pedestrian, traffic_cone, construction_vehicle, trailer

## 📊 Estrutura dos Dados

```json
{
  "meta": {
    "dataset": "NuScenesDataset",
    "config": "bevdet-multi-map-aug-seg-only-6class-hsda.py"
  },
  "results": {
    "sample_token_1": [
      {
        "bbox_3d": [...],  // Coordenadas 3D
        "score_3d": 0.xx,  // Confiança
        "label_3d": x      // Classe
      }
    ]
  }
}
```

## 🎯 Características

### Dados Incluídos:
- ✅ **Bounding boxes 3D** - Coordenadas dos objetos detectados
- ✅ **Scores de confiança** - Probabilidade de cada detecção
- ✅ **Labels de classe** - Tipo de objeto detectado
- ✅ **Sample tokens** - Identificadores únicos das amostras

### Dados NÃO Incluídos:
- ❌ **Métricas mIoU** - Estão em `results/metrics/`
- ❌ **Segmentação BEV** - Resultados de pixel-level segmentation
- ❌ **Mapas de calor** - Confidence maps
- ❌ **Ground truth** - Labels verdadeiros

## 📈 Como Usar

### Carregar resultados:
```python
import json

with open('results_vis.json', 'r') as f:
    data = json.load(f)

print(f"Total de amostras: {len(data['results'])}")
print(f"Primeira amostra: {list(data['results'].keys())[0]}")
```

### Analisar detecções:
```python
# Contar detecções por amostra
for token, detections in data['results'].items():
    print(f"Sample {token}: {len(detections)} objetos detectados")
```

## 🔗 Arquivos Relacionados

- **Métricas**: `../metrics/original_miou_metrics.json`
- **Visualizações**: `../visualizations_all/*.jpg`
- **Scripts**: `../../scripts/evaluation/`

## ⚠️ Importante

Este arquivo contém apenas **resultados de detecção de objetos** (bbox). Para métricas de segmentação BEV (mIoU), consulte a pasta `results/metrics/`.

---
**Gerado em:** Data da inferência  
**Modelo:** HSDA BEVDet epoch_20.pth  
**Dataset:** nuScenes mini (81 amostras)
