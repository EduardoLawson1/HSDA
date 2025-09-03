# 📋 Project Reports

Esta pasta contém relatórios e documentação detalhada do projeto HSDA.

## 📁 Relatórios Disponíveis

### Métricas e Resultados
- **`RELATORIO_METRICAS_MIOU.md`** - Relatório completo das métricas mIoU
- **`RELATORIO_METRICAS_NUMERICAS.md`** - Análise numérica detalhada
- **`FINAL_RESULTS.md`** - Resultados finais do projeto

### Documentação do Projeto
- **`FINAL_PROJECT_CHECK.md`** - Checklist final de conclusão

## 📊 Destaques dos Resultados

### mIoU Global: 0.150
- Calculado com código original do HSDA
- 6 classes de segmentação BEV
- Dataset nuScenes (81 amostras)

### Performance por Classe:
1. **drivable_area**: 0.298 (melhor)
2. **divider**: 0.197 
3. **walkway**: 0.143
4. **carpark_area**: 0.117
5. **ped_crossing**: 0.098
6. **stop_line**: 0.048 (pior)

## 📈 Análises Incluídas

- Comparação de performance entre classes
- Análise por threshold (0.35 - 0.65)
- Identificação de pontos fortes e fracos
- Sugestões de melhorias

## 🔗 Referências

Para dados brutos consulte:
- `results/metrics/` - Arquivos de métricas
- `scripts/evaluation/` - Scripts utilizados

---
**Última atualização:** 3 de setembro de 2025  
**Status:** Completo ✅
