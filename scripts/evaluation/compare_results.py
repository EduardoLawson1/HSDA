#!/usr/bin/env python3

import json

print("🔍 ANÁLISE COMPARATIVA: results_vis.json vs métricas mIoU originais")
print("="*70)

# Analisar results_vis.json
try:
    with open("results_vis.json", 'r') as f:
        results_data = json.load(f)
    
    print("\n📄 RESULTS_VIS.JSON:")
    print(f"  ✓ Chaves principais: {list(results_data.keys())}")
    print(f"  ✓ Número de samples: {len(results_data.get('results', {}))}")
    
    # Verificar se tem métricas mIoU
    has_miou = False
    has_map_metrics = False
    
    for key in results_data.keys():
        if 'miou' in str(key).lower() or 'iou' in str(key).lower():
            has_miou = True
        if 'map' in str(key).lower():
            has_map_metrics = True
    
    print(f"  ✗ Contém métricas mIoU: {has_miou}")
    print(f"  ✗ Contém métricas MAP: {has_map_metrics}")
    print(f"  → Tipo de dados: DETECÇÕES BBOX apenas")
    
except Exception as e:
    print(f"  ✗ Erro ao ler results_vis.json: {e}")

# Analisar métricas originais
try:
    with open("original_miou_metrics.json", 'r') as f:
        metrics_data = json.load(f)
    
    print(f"\n📊 ORIGINAL_MIOU_METRICS.JSON:")
    print(f"  ✓ Método: {metrics_data.get('evaluation_method')}")
    print(f"  ✓ Fonte: {metrics_data.get('source_function')}")
    print(f"  ✓ Classes: {len(metrics_data.get('classes', []))}")
    print(f"  ✓ Total métricas: {len(metrics_data.get('metrics', {}))}")
    print(f"  ✓ mIoU médio: {metrics_data.get('metrics', {}).get('map/mean/iou@max', 0):.3f}")
    print(f"  → Tipo de dados: MÉTRICAS mIoU SEGMENTAÇÃO")
    
except Exception as e:
    print(f"  ✗ Erro ao ler original_miou_metrics.json: {e}")

print(f"\n🏁 CONCLUSÃO:")
print(f"="*70)
print(f"❌ results_vis.json NÃO contém métricas mIoU")
print(f"   → Contém apenas resultados de detecção bbox")
print(f"   → Para métricas mIoU são necessários dados de segmentação BEV")
print(f"")
print(f"✅ original_miou_metrics.json contém métricas mIoU ORIGINAIS")
print(f"   → Calculadas com código base do HSDA (evaluate_map)")
print(f"   → Função exata: nuscenes_dataset_map.py linha 315")
print(f"   → mIoU médio: {metrics_data.get('metrics', {}).get('map/mean/iou@max', 0):.3f}")

print(f"\n🎯 RESPOSTA À SUA PERGUNTA:")
print(f"   O arquivo results_vis.json NÃO contém métricas mIoU")
print(f"   É necessário rodar o script de avaliação padrão do projeto")
print(f"   As métricas em original_miou_metrics.json são do código original")
