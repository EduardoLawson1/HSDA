# Localização dos Cálculos de Métricas mIoU no HSDA

## 📍 **Onde Encontrar os Cálculos de mIoU**

### 1. **mIoU para Segmentação 3D Geral** 
📁 **Arquivo**: `mmdet3d/core/evaluation/seg_eval.py`

```python
def seg_eval(gt_labels, seg_preds, label2cat, ignore_index, logger=None):
    """Semantic Segmentation Evaluation - FUNÇÃO PRINCIPAL DE mIoU"""
    
    # Linha 102: Cálculo do mIoU
    iou = per_class_iou(sum(hist_list))
    miou = np.nanmean(iou)  # ← CÁLCULO PRINCIPAL DO mIoU
    
    # Linhas 116-120: Retorno das métricas
    ret_dict['miou'] = float(miou)
    ret_dict['acc'] = float(acc)
    ret_dict['acc_cls'] = float(acc_cls)
```

**Funções auxiliares importantes:**
- `fast_hist()` - Matriz de confusão (linhas 7-25)
- `per_class_iou()` - IoU por classe (linhas 28-38)
- `get_acc()` - Accuracy geral (linhas 41-52)

### 2. **mIoU Específico para Mapas BEV do HSDA**
📁 **Arquivo**: `mmdet3d_plugin/datasets/nuscenes_dataset_map.py`

```python
def evaluate_map(self, results):
    """Avaliação específica para segmentação de mapas BEV - HSDA"""
    
    # Linhas 310-312: Configuração
    thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    num_classes = len(self.map_classes)
    
    # Linhas 313-315: Inicialização das métricas
    tp = torch.zeros(num_classes, num_thresholds)  # True Positives
    fp = torch.zeros(num_classes, num_thresholds)  # False Positives  
    fn = torch.zeros(num_classes, num_thresholds)  # False Negatives
    
    # Linhas 317-332: Loop principal de cálculo
    for result in results:
        pred = result["bev_seg"]      # Predições do modelo
        label = result["gt_masks_bev"] # Ground truth
        
        # Cálculo de TP, FP, FN para cada threshold
        tp += (pred & label).sum(dim=1)
        fp += (pred & ~label).sum(dim=1)
        fn += (~pred & label).sum(dim=1)
    
    # Linha 333: CÁLCULO DO IoU
    ious = tp / (tp + fp + fn + 1e-7)  # ← FÓRMULA IoU PRINCIPAL
    
    # Linha 340: mIoU médio
    metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
```

### 3. **Classes de Mapa Avaliadas no HSDA**

📁 **Arquivo**: `configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py`

```python
# Linhas 41-42: Classes de mapa para segmentação BEV
grid_config = {
    'map_classes': [
        'drivable_area',      # Área dirigível
        'ped_crossing',       # Faixa de pedestres  
        'walkway',            # Calçada
        'stop_line',          # Linha de parada
        'carpark_area',       # Área de estacionamento
        'divider'             # Divisor de pista
    ]
}
```

## 🔄 **Fluxo de Execução das Métricas mIoU**

### **Passo 1: Durante o Teste**
```python
# tools/test.py - linha 221
print(dataset.evaluate(outputs, **eval_kwargs))
```

### **Passo 2: Método evaluate() Chamado**
```python
# mmdet3d_plugin/datasets/nuscenes_dataset_map.py - linha 343
def evaluate(self, results, metric='bbox', ...):
    if 'map' in metric:
        results_dict.update(self.evaluate_map(map_list))  # ← Chama cálculo mIoU
```

### **Passo 3: Cálculo Real do mIoU**
```python
# mmdet3d_plugin/datasets/nuscenes_dataset_map.py - linha 309
def evaluate_map(self, results):
    # Aqui acontece o cálculo real do IoU/mIoU
```

## 📊 **Tipos de Métricas mIoU Retornadas**

### **1. IoU por Classe e Threshold**
```python
# Exemplo de saída:
{
    "map/drivable_area/iou@0.35": 0.7234,
    "map/drivable_area/iou@0.40": 0.6892,
    "map/drivable_area/iou@0.45": 0.6543,
    # ... para cada classe e threshold
}
```

### **2. IoU Máximo por Classe**
```python
{
    "map/drivable_area/iou@max": 0.7234,
    "map/ped_crossing/iou@max": 0.5123,
    "map/walkway/iou@max": 0.6789,
    # ... para todas as classes
}
```

### **3. mIoU Médio Geral**
```python
{
    "map/mean/iou@max": 0.6421  # ← MÉTRICA PRINCIPAL mIoU
}
```

## 🎯 **Como Executar Avaliação com mIoU**

### **Comando para Teste com Métricas de Mapa:**
```bash
python tools/test.py \
    configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
    checkpoints/epoch_20.pth \
    --eval=bboxmap \           # ← Inclui métricas de bbox + mapa
    --out results.pkl
```

### **Somente Métricas de Mapa:**
```bash
python tools/test.py \
    configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
    checkpoints/epoch_20.pth \
    --eval=map \              # ← Somente métricas de segmentação
    --out results.pkl
```

## 🧮 **Fórmula Matemática do IoU**

### **IoU (Intersection over Union):**
```
IoU = TP / (TP + FP + FN)

Onde:
- TP = True Positives (pixels corretamente classificados como positivos)
- FP = False Positives (pixels incorretamente classificados como positivos)  
- FN = False Negatives (pixels incorretamente classificados como negativos)
```

### **mIoU (mean Intersection over Union):**
```
mIoU = (1/N) * Σ(IoU_i)

Onde:
- N = número de classes
- IoU_i = IoU da classe i
```

## 📁 **Estrutura de Arquivos das Métricas mIoU**

```
HSDA/
├── mmdet3d/core/evaluation/
│   └── seg_eval.py                    # ← mIoU geral para segmentação 3D
├── mmdet3d_plugin/datasets/
│   └── nuscenes_dataset_map.py        # ← mIoU específico para mapas BEV HSDA
├── mmdet3d/datasets/
│   └── custom_3d_seg.py               # ← Base para datasets de segmentação
└── tools/
    └── test.py                        # ← Script que executa avaliação
```

## 🔍 **Depuração e Debugging**

### **Para Ver Cálculo Detalhado:**
```python
# Adicionar prints em nuscenes_dataset_map.py, linha 333
print(f"TP shape: {tp.shape}, FP shape: {fp.shape}, FN shape: {fn.shape}")
print(f"IoUs por classe: {ious}")
print(f"mIoU final: {ious.max(dim=1).values.mean().item()}")
```

### **Para Verificar Classes de Mapa:**
```python
# Em nuscenes_dataset_map.py
print(f"Classes de mapa: {self.map_classes}")
print(f"Número de classes: {len(self.map_classes)}")
```

## ✅ **Resumo das Localizações**

| **Tipo de mIoU** | **Arquivo** | **Função** | **Linha** |
|-------------------|-------------|------------|-----------|
| **mIoU Geral 3D** | `mmdet3d/core/evaluation/seg_eval.py` | `seg_eval()` | 102 |
| **mIoU Mapas BEV HSDA** | `mmdet3d_plugin/datasets/nuscenes_dataset_map.py` | `evaluate_map()` | 333 |
| **Execução de Teste** | `tools/test.py` | `main()` | 221 |
| **Configuração Classes** | `configs/bevdet_hsda/...` | `grid_config` | 41-42 |

---

## 🎯 **Comando Rápido para Testar mIoU:**

```bash
# Testar mIoU do HSDA
./run_evaluation.sh  # Usa script automatizado criado

# Ou comando direto:
sudo docker exec -it hsda_container bash -c "
cd /mmdetection3d && 
python tools/test.py \
    configs/bevdet_hsda/bevdet-multi-map-aug-seg-only-6class-hsda.py \
    checkpoints/epoch_20.pth \
    --eval=bboxmap \
    --out results_miou.pkl
"
```

**As métricas mIoU aparecerão no terminal e serão salvas nos arquivos de resultado!** 🎉
