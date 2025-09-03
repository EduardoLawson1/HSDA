# 📊 Metrics Results

Esta pasta contém os resultados das métricas de avaliação do modelo HSDA.

## 📁 Arquivos Disponíveis

### Métricas mIoU
- **`original_miou_metrics.json`** - Métricas completas calculadas com código original do HSDA
- **`metricas_miou_hsda.csv`** - Métricas em formato CSV para análise em planilhas
- **`segmentation_metrics.json`** - Métricas de segmentação detalhadas
- **`metrics_summary.json`** - Resumo geral das métricas

## 📈 Principais Resultados

### mIoU por Classe (valores máximos):
- **drivable_area**: 0.298 🟢
- **divider**: 0.197 🟡  
- **walkway**: 0.143 🟡
- **carpark_area**: 0.117 🟠
- **ped_crossing**: 0.098 🔴
- **stop_line**: 0.048 🔴

### Média Global:
- **mIoU Médio**: 0.150

## 🔬 Autenticidade

Todas as métricas foram calculadas usando o código original do projeto HSDA:
- Função: `nuscenes_dataset_map.py:evaluate_map()` linha 315
- Thresholds: [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
- Dataset: 81 amostras do nuScenes

## 📋 Como Usar

1. **Para análises**: Use `metricas_miou_hsda.csv` em Excel/Google Sheets
2. **Para desenvolvimento**: Use `original_miou_metrics.json` com todos os detalhes
3. **Para relatórios**: Consulte os arquivos em `docs/reports/`

---
**Gerado em:** 3 de setembro de 2025  
**Método:** Código Original HSDA
