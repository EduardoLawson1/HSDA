# 📊 Results Directory

Esta pasta contém todos os resultados gerados pelo modelo HSDA organizados por tipo.

## 📁 Estrutura Organizacional

### `/metrics/` - Métricas de Avaliação
Métricas calculadas com código original do HSDA:
- `original_miou_metrics.json` - Métricas mIoU completas 
- `metricas_miou_hsda.csv` - Formato CSV para planilhas
- `segmentation_metrics.json` - Métricas de segmentação
- `metrics_summary.json` - Resumo geral

### `/inference/` - Resultados de Inferência
Outputs brutos do modelo:
- `results_vis.json` - Resultados de detecção bbox (487KB)
- Contém 81 amostras com detecções de objetos

### `/visualizations_all/` - Visualizações
Imagens com predições do modelo:
- 81 arquivos .jpg com visualizações BEV
- Resultados de segmentação sobrepostos

### `/results_evaluation_*/` - Logs de Avaliação
Logs e outputs de execuções de avaliação.

## 🎯 Principais Resultados

### mIoU Segmentação BEV:
| Classe | mIoU@max | Performance |
|--------|----------|-------------|
| drivable_area | 0.298 | 🟢 Melhor |
| divider | 0.197 | 🟡 Boa |
| walkway | 0.143 | 🟡 Moderada |
| carpark_area | 0.117 | 🟠 Baixa |
| ped_crossing | 0.098 | 🔴 Crítica |
| stop_line | 0.048 | 🔴 Crítica |

**mIoU Médio Global: 0.150**

### Detecção de Objetos:
- 81 amostras processadas
- Resultados em `inference/results_vis.json`
- Classes: car, truck, bus, barrier, etc.

## 🔍 Como Usar

### Para análise de métricas:
```bash
cd results/metrics/
# CSV para planilhas
cat metricas_miou_hsda.csv
# JSON completo
cat original_miou_metrics.json
```

### Para resultados de inferência:
```bash
cd results/inference/
# Visualizar detecções
python -c "import json; print(len(json.load(open('results_vis.json'))['results']))"
```

### Para visualizações:
```bash
cd results/visualizations_all/
ls *.jpg | head -5  # Ver primeiras 5 imagens
```

## 📝 Tipos de Dados

### Métricas (`/metrics/`):
- **Tipo**: Avaliação quantitativa
- **Origem**: Código original HSDA
- **Formato**: JSON, CSV
- **Uso**: Análise de performance

### Inferência (`/inference/`):
- **Tipo**: Detecções bbox
- **Origem**: Modelo HSDA treinado
- **Formato**: JSON estruturado
- **Uso**: Resultados brutos

### Visualizações (`/visualizations_all/`):
- **Tipo**: Imagens BEV
- **Origem**: Predições + ground truth
- **Formato**: JPG
- **Uso**: Análise visual

---
**Organizado em:** 3 de setembro de 2025  
**Status:** Completo e Organizado ✅
