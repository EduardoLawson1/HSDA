# Onde o Modelo HSDA Retorna as Métricas

## Resumo Executivo

O modelo HSDA retorna as métricas de avaliação em **múltiplos locais** durante o processo de teste/avaliação. Este documento explica onde encontrar cada tipo de métrica.

## 1. Locais de Saída das Métricas

### 📊 **Terminal/Console (Saída Principal)**
```bash
# Comando de teste padrão
python tools/test.py \
    configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
    checkpoints/epoch_20.pth \
    --eval=bbox \
    --out results_hsda.pkl
```

**Métricas exibidas no terminal:**
- **mAP (mean Average Precision)**: Precisão média geral
- **NDS (NuScenes Detection Score)**: Score específico do NuScenes
- **AP por classe**: Precisão média para cada classe de objeto
- **Erros de TP (True Positive)**: Erros de localização, orientação, velocidade, etc.

### 📁 **Arquivo PKL (Resultados Brutos)**
```bash
# Especificado pelo parâmetro --out
results_hsda.pkl
```
**Conteúdo:**
- Predições brutas para cada amostra
- Bounding boxes 3D detectadas
- Scores de confiança
- Labels preditas

### 📈 **Arquivo JSON (Métricas Detalhadas)**
```bash
# Gerado automaticamente em:
work_dirs/[experiment_name]/[timestamp]/metrics_summary.json
```

## 2. Estrutura das Métricas no NuScenes

### **Métricas Principais**
```json
{
    "mean_ap": 0.XXXX,           // mAP geral
    "nd_score": 0.XXXX,          // NuScenes Detection Score
    "label_aps": {               // AP por classe
        "car": {
            "0.5": 0.XXXX,       // AP@0.5m
            "1.0": 0.XXXX,       // AP@1.0m
            "2.0": 0.XXXX,       // AP@2.0m
            "4.0": 0.XXXX        // AP@4.0m
        },
        "truck": {...},
        "pedestrian": {...}
    },
    "tp_errors": {               // Erros de True Positives
        "trans_err": 0.XXXX,     // Erro de translação
        "scale_err": 0.XXXX,     // Erro de escala
        "orient_err": 0.XXXX,    // Erro de orientação
        "vel_err": 0.XXXX,       // Erro de velocidade
        "attr_err": 0.XXXX       // Erro de atributos
    }
}
```

## 3. Código Específico Onde Métricas São Calculadas

### **No arquivo tools/test.py (linhas 216-221):**
```python
if args.eval:
    eval_kwargs = cfg.get('evaluation', {}).copy()
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    print(dataset.evaluate(outputs, **eval_kwargs))  # ← MÉTRICAS IMPRESSAS AQUI
```

### **No arquivo mmdet3d/datasets/nuscenes_dataset.py (linhas 483-493):**
```python
# record metrics
metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
detail = dict()
metric_prefix = f'{result_name}_NuScenes'
for name in self.CLASSES:
    for k, v in metrics['label_aps'][name].items():
        val = float('{:.4f}'.format(v))
        detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val

detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
```

## 4. Exemplo de Saída de Métricas

### **Terminal Output Típico:**
```
Evaluating bboxes of pts_bbox
+----------+--------+--------+--------+--------+--------+
| AP@0.5   | AP@1.0 | AP@2.0 | AP@4.0 | mAP    | NDS    |
+----------+--------+--------+--------+--------+--------+
| 0.2534   | 0.3421 | 0.4123 | 0.4567 | 0.3661 | 0.4234 |
+----------+--------+--------+--------+--------+--------+

Per-class results:
+------------------+--------+--------+--------+--------+
| Class            | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 |
+------------------+--------+--------+--------+--------+
| car              | 0.7234 | 0.8123 | 0.8567 | 0.8901 |
| truck            | 0.4123 | 0.5234 | 0.6123 | 0.6789 |
| pedestrian       | 0.1234 | 0.2345 | 0.3456 | 0.4567 |
+------------------+--------+--------+--------+--------+
```

## 5. Como Salvar e Acessar Métricas

### **Método 1: Redirecionamento de Terminal**
```bash
python tools/test.py \
    configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
    checkpoints/epoch_20.pth \
    --eval=bbox \
    --out results_hsda.pkl > metricas_resultado.txt 2>&1
```

### **Método 2: Parsing do JSON**
```python
import json
import mmcv

# Carregar métricas detalhadas
metrics = mmcv.load('work_dirs/experiment/metrics_summary.json')
print(f"mAP: {metrics['mean_ap']:.4f}")
print(f"NDS: {metrics['nd_score']:.4f}")

# Métricas por classe
for class_name, class_metrics in metrics['label_aps'].items():
    print(f"{class_name} AP@2.0: {class_metrics['2.0']:.4f}")
```

### **Método 3: Log Files**
Verificar logs em:
```
work_dirs/[experiment_name]/[timestamp].log
```

## 6. Configuração de Métricas

### **No arquivo de configuração:**
```python
# Em configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py
evaluation = dict(
    interval=20,           # Avaliar a cada 20 epochs
    pipeline=eval_pipeline # Pipeline de avaliação
)
```

### **Métricas Disponíveis:**
- `bbox` - Métricas de bounding box 3D
- `bboxmap` - Métricas de bbox + mapas (se disponível)
- `mAP` - Mean Average Precision
- `NDS` - NuScenes Detection Score

## 7. Interpretação das Métricas

### **mAP (Mean Average Precision)**
- **Faixa**: 0.0 - 1.0 (quanto maior, melhor)
- **Significado**: Precisão média em diferentes thresholds de distância
- **Bom resultado**: > 0.4 para modelos de referência

### **NDS (NuScenes Detection Score)**
- **Faixa**: 0.0 - 1.0 (quanto maior, melhor)
- **Significado**: Score composto considerando detecção + erros
- **Bom resultado**: > 0.45 para modelos competitivos

### **AP por Distância**
- **AP@0.5m**: Detecções muito precisas
- **AP@1.0m**: Detecções precisas
- **AP@2.0m**: Detecções moderadamente precisas
- **AP@4.0m**: Detecções com tolerância maior

## 8. Troubleshooting Comum

### **Problema: Métricas não aparecem**
```bash
# Verificar se --eval está especificado
python tools/test.py config.py checkpoint.pth --eval=bbox
```

### **Problema: Arquivo de métricas não encontrado**
```bash
# Verificar se evaluation está configurado no config
evaluation = dict(interval=20, pipeline=eval_pipeline)
```

### **Problema: Métricas muito baixas**
- ✅ Verificar se o modelo foi treinado adequadamente
- ✅ Verificar se os dados de teste estão corretos
- ✅ Verificar se as transformações de coordenadas estão corretas

## 9. Scripts de Automação

### **Script para Extrair Métricas:**
```bash
#!/bin/bash
# run_evaluation.sh

CONFIG="configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py"
CHECKPOINT="checkpoints/epoch_20.pth"
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR

python tools/test.py $CONFIG $CHECKPOINT \
    --eval=bbox \
    --out $OUTPUT_DIR/predictions.pkl \
    > $OUTPUT_DIR/metrics.txt 2>&1

echo "Métricas salvas em: $OUTPUT_DIR/metrics.txt"
```

---

## ✅ Resumo Final

**As métricas do modelo HSDA são retornadas em:**

1. **🖥️ Terminal** - Exibição imediata durante teste
2. **📁 arquivo.pkl** - Predições brutas (parâmetro `--out`)
3. **📊 metrics_summary.json** - Métricas detalhadas estruturadas
4. **📝 Log files** - Logs completos do experimento

**Para acessar rapidamente:**
```bash
# Ver métricas no terminal
python tools/test.py config.py checkpoint.pth --eval=bbox

# Salvar métricas em arquivo
python tools/test.py config.py checkpoint.pth --eval=bbox > metrics.txt
```
