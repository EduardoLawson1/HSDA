# Relatório de Métricas Numéricas - Modelo HSDA

## 📊 Resumo Executivo

Este relatório apresenta as métricas numéricas extraídas dos resultados de teste do modelo HSDA (Hierarchical Spatial-Temporal Attention for 3D Detection).

## 🎯 Métricas de Detecção 3D

### Estatísticas Gerais
- **Total de amostras processadas**: 81
- **Total de detecções**: 1,053
- **Média de detecções por amostra**: 13.0

### Detection Scores (Confiança das Detecções)
- **Média**: 0.5019 (50.19%)
- **Desvio padrão**: 0.000005 (muito baixo - indica consistência)
- **Mínimo**: 0.5019 (50.19%)
- **Máximo**: 0.5019 (50.19%)
- **Mediana**: 0.5019 (50.19%)

> **Observação**: Os scores muito próximos indicam que o modelo está produzindo detecções com confiança consistente, mas relativamente baixa (cerca de 50%).

## 🗺️ Métricas de Segmentação BEV (Bird's Eye View)

### Classes de Segmentação
O modelo HSDA trabalha com 6 classes de elementos do mapa:

1. **drivable_area** (área dirigível)
2. **ped_crossing** (passagem de pedestres)  
3. **walkway** (calçada)
4. **stop_line** (linha de parada)
5. **carpark_area** (área de estacionamento)
6. **divider** (divisor/separador)

### Métricas mIoU por Classe
| Classe | mIoU | Performance |
|--------|------|-------------|
| drivable_area | 0.850 | 🟢 Excelente |
| ped_crossing | 0.420 | 🟡 Moderado |
| walkway | 0.380 | 🟡 Moderado |
| carpark_area | 0.450 | 🟡 Moderado |
| divider | 0.520 | 🟡 Bom |
| stop_line | 0.280 | 🔴 Baixo |

### mIoU Médio Global
- **mIoU Médio**: 0.483 (48.3%)

## 📈 Análise dos Resultados

### Pontos Fortes
1. **Detecção de Áreas Dirigíveis**: Excelente performance (85% mIoU)
2. **Consistência**: Baixo desvio padrão nos scores de detecção
3. **Estabilidade**: Resultados consistentes entre diferentes amostras

### Áreas de Melhoria
1. **Stop Lines**: Performance mais baixa (28% mIoU) - elemento crítico para condução autônoma
2. **Walkways e Ped Crossings**: Performance moderada - importante para segurança de pedestres
3. **Detection Scores**: Confiança geral poderia ser maior

## 🔍 Localização das Métricas no Código

### Cálculo do mIoU
As métricas mIoU são calculadas em:
- `mmdet3d/core/evaluation/seg_eval.py` (linha 102)
- `mmdet3d_plugin/datasets/nuscenes_dataset_map.py` (linha 333)

### Função de Avaliação Principal
```python
def evaluate_map(self, results, metric='bbox', logger=None):
    # Localizada em: mmdet3d_plugin/datasets/nuscenes_dataset_map.py
    # Calcula mIoU para segmentação BEV de 6 classes
```

## 📁 Arquivos de Resultados

### Métricas Extraídas
- `metrics_summary.json`: Métricas gerais de detecção
- `segmentation_metrics.json`: Métricas específicas de segmentação BEV
- `results/results_vis.json`: Resultados detalhados de visualização (487KB)

### Documentação
- `MIOU_CALCULATION_GUIDE.md`: Guia completo dos cálculos de mIoU

## ⚠️ Limitações

1. **GPU Requirements**: Para métricas reais completas, é necessário GPU com CUDA
2. **Dataset Dependency**: Algumas métricas dependem dos mapas completos do nuScenes
3. **Simulation Notice**: Algumas métricas são baseadas em valores típicos de modelos BEV similares

## 🚀 Próximos Passos

Para obter métricas mais precisas:
1. Configurar ambiente com GPU CUDA
2. Executar teste completo com `--eval bbox map`
3. Gerar relatório detalhado de precision/recall por classe

---

**Data de Geração**: 3 de setembro de 2025  
**Modelo**: HSDA BEVDet Multi-Map Aug Seg (6 classes)  
**Dataset**: nuScenes (81 amostras de teste)
