# 🔬 Evaluation Scripts

Esta pasta contém scripts para avaliação e cálculo de métricas do modelo HSDA.

## 📁 Scripts Disponíveis

### Calculadores de Métricas
- **`original_miou_calculator.py`** - Calcula mIoU usando código original do HSDA
- **`extract_metrics.py`** - Extrai métricas de arquivos de resultados
- **`compare_results.py`** - Compara diferentes tipos de resultados

### Scripts de Execução
- **`run_eval_original.sh`** - Executa avaliação original do HSDA
- **`run_evaluation.sh`** - Script de avaliação geral
- **`run_inference.sh`** - Executa inferência do modelo
- **`run_simple_test.sh`** - Teste simples de funcionamento

## 🚀 Como Usar

### Para calcular métricas mIoU:
```bash
python original_miou_calculator.py
```

### Para executar avaliação completa:
```bash
./run_evaluation.sh
```

### Para fazer inferência:
```bash
./run_inference.sh
```

## ⚙️ Requisitos

- Docker container: `hsda_container`
- Ambiente: MMDetection3D com PyTorch
- Dataset: nuScenes preparado

## 📊 Outputs

Os scripts geram resultados em:
- `results/metrics/` - Arquivos de métricas
- `results/` - Arquivos de resultados de inferência
- `work_dirs/` - Logs e checkpoints

## 🔍 Troubleshooting

Se encontrar erros:
1. Verifique se o container Docker está rodando
2. Confirme que os dados estão montados corretamente
3. Verifique dependências no container

---
**Criado em:** 3 de setembro de 2025  
**Compatível com:** HSDA v1.0
