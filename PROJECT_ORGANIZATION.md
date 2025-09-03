# 📂 HSDA Project Structure

Organização atualizada dos arquivos do projeto HSDA para melhor manutenção e versionamento.

## 🗂️ Nova Estrutura Organizacional

### 📊 `/results/`
**Diretório principal de resultados**

#### `/results/metrics/`
**Métricas de avaliação**
- `original_miou_metrics.json` - Métricas mIoU originais (JSON completo)
- `metricas_miou_hsda.csv` - Métricas em formato CSV  
- `segmentation_metrics.json` - Métricas de segmentação
- `metrics_summary.json` - Resumo das métricas
- `README.md` - Documentação dos resultados

#### `/results/inference/`
**Resultados de inferência**
- `results_vis.json` - Resultados de detecção bbox (487KB)
- `README.md` - Documentação dos resultados de inferência

#### `/results/visualizations_all/`
**Visualizações BEV**
- 81 imagens .jpg com predições sobrepostas

### 🔬 `/scripts/evaluation/`
**Scripts de avaliação e cálculo de métricas**
- `original_miou_calculator.py` - Calculador de mIoU original
- `extract_metrics.py` - Extrator de métricas
- `compare_results.py` - Comparador de resultados
- `run_eval_original.sh` - Script de avaliação original
- `run_evaluation.sh` - Script de avaliação geral
- `run_inference.sh` - Script de inferência
- `run_simple_test.sh` - Teste simples
- `README.md` - Documentação dos scripts

### 📋 `/docs/reports/`
**Relatórios e documentação detalhada**
- `RELATORIO_METRICAS_MIOU.md` - Relatório completo mIoU
- `RELATORIO_METRICAS_NUMERICAS.md` - Análise numérica
- `FINAL_RESULTS.md` - Resultados finais
- `FINAL_PROJECT_CHECK.md` - Checklist do projeto
- `README.md` - Índice dos relatórios

## 🎯 Principais Resultados Obtidos

### mIoU por Classe (máximo):
| Classe | mIoU | Status |
|--------|------|--------|
| drivable_area | 0.298 | 🟢 Boa |
| divider | 0.197 | 🟡 Moderada |
| walkway | 0.143 | 🟡 Moderada |
| carpark_area | 0.117 | 🟠 Baixa |
| ped_crossing | 0.098 | 🔴 Crítica |
| stop_line | 0.048 | 🔴 Crítica |

**mIoU Médio Global: 0.150**

## ✅ Arquivos Organizados

### Movidos para `results/metrics/`:
- ✅ original_miou_metrics.json
- ✅ metricas_miou_hsda.csv  
- ✅ segmentation_metrics.json
- ✅ metrics_summary.json

### Movidos para `results/inference/`:
- ✅ results_vis.json

### Movidos para `scripts/evaluation/`:
- ✅ original_miou_calculator.py
- ✅ extract_metrics.py
- ✅ compare_results.py
- ✅ run_eval_original.sh
- ✅ run_evaluation.sh
- ✅ run_inference.sh
- ✅ run_simple_test.sh

### Movidos para `docs/reports/`:
- ✅ RELATORIO_METRICAS_MIOU.md
- ✅ RELATORIO_METRICAS_NUMERICAS.md
- ✅ FINAL_RESULTS.md
- ✅ FINAL_PROJECT_CHECK.md

## 🔄 Próximos Passos

1. **Commit das mudanças** no repositório GitHub
2. **Update do README principal** com nova estrutura
3. **Documentação adicional** se necessário

## 📝 Como Acessar

### Para métricas:
```bash
cd results/metrics/
cat original_miou_metrics.json
```

### Para executar avaliação:
```bash
cd scripts/evaluation/
python original_miou_calculator.py
```

### Para relatórios:
```bash
cd docs/reports/
cat RELATORIO_METRICAS_MIOU.md
```

---
**Organizado em:** 3 de setembro de 2025  
**Status:** Pronto para commit ✅
