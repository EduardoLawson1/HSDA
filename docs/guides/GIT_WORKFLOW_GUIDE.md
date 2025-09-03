# Guia de Workflow Git para HSDA

## Situação Atual do Repositório

- **Repositório**: EduardoLawson1/HSDA 
- **Status**: Atualizado e limpo
- **Branch principal**: main
- **Último commit**: Complete HSDA project improvements and documentation

## 1. Fluxo de Trabalho Recomendado

### Para Pequenas Atualizações
```bash
# 1. Sempre começar com o repositório atualizado
git pull origin main

# 2. Fazer suas modificações nos arquivos

# 3. Adicionar arquivos modificados
git add .
# ou adicionar arquivos específicos:
git add arquivo_especifico.py

# 4. Fazer commit com mensagem descritiva
git commit -m "feat: descrição da mudança"

# 5. Enviar para o GitHub
git push origin main
```

### Para Funcionalidades Maiores (Recomendado)
```bash
# 1. Criar uma nova branch para a funcionalidade
git checkout -b nova-funcionalidade

# 2. Fazer suas modificações

# 3. Adicionar e comitar
git add .
git commit -m "feat: nova funcionalidade implementada"

# 4. Enviar a branch para o GitHub
git push origin nova-funcionalidade

# 5. Criar Pull Request no GitHub
# 6. Após aprovação, fazer merge
```

## 2. Convenções de Commit

Use prefixos semânticos nas mensagens de commit:

- `feat:` - Nova funcionalidade
- `fix:` - Correção de bug
- `docs:` - Mudanças na documentação
- `style:` - Formatação, espaços, etc.
- `refactor:` - Refatoração de código
- `test:` - Adição ou modificação de testes
- `chore:` - Tarefas de manutenção

Exemplos:
```bash
git commit -m "feat: add new visualization script"
git commit -m "fix: correct bounding box coordinates"
git commit -m "docs: update README with new instructions"
```

## 3. Comandos Essenciais

### Verificar Status
```bash
git status                  # Ver arquivos modificados
git log --oneline -10       # Ver últimos 10 commits
git diff                    # Ver mudanças não commitadas
```

### Gerenciar Mudanças
```bash
git add arquivo.py          # Adicionar arquivo específico
git add .                   # Adicionar todos os arquivos
git commit -m "mensagem"    # Fazer commit
git push origin main        # Enviar para GitHub
```

### Sincronizar com Repositório Remoto
```bash
git pull origin main        # Baixar últimas mudanças
git fetch                   # Baixar refs sem fazer merge
git merge origin/main       # Fazer merge manual
```

### Gerenciar Branches
```bash
git branch                  # Listar branches
git checkout nova-branch    # Trocar para branch
git checkout -b nova-branch # Criar e trocar para nova branch
git merge feature-branch    # Fazer merge de branch
git branch -d feature-branch # Deletar branch local
```

## 4. Situações Específicas do HSDA

### Atualizar Documentação
```bash
# Para atualizações de documentação
git add README.md QUICK_START.md
git commit -m "docs: update project documentation"
git push origin main
```

### Adicionar Novos Scripts
```bash
# Para novos scripts ou ferramentas
git add tools/novo_script.py
git commit -m "feat: add new inference automation script"
git push origin main
```

### Correções de Bugs
```bash
# Para correções
git add mmdet3d_plugin/models/dense_heads/
git commit -m "fix: correct coordinate transformation in BEV detection"
git push origin main
```

## 5. Boas Práticas

1. **Sempre fazer pull antes de começar trabalho novo**
2. **Usar branches para funcionalidades grandes**
3. **Commits pequenos e frequentes são melhores**
4. **Mensagens de commit descritivas**
5. **Testar código antes de fazer push**
6. **Revisar mudanças com `git diff` antes de comitar**

## 6. Colaboração em Equipe

### Se outros membros estão trabalhando:
```bash
# 1. Sempre sincronizar antes de começar
git pull origin main

# 2. Se houver conflitos, resolver manualmente
# 3. Usar branches para evitar conflitos

# 4. Comunicar mudanças grandes para a equipe
```

### Para resolver conflitos:
```bash
# 1. Git vai marcar arquivos com conflitos
git pull origin main  # Se houver conflitos

# 2. Editar arquivos marcados, escolher versão correta
# 3. Adicionar arquivos resolvidos
git add arquivo_resolvido.py

# 4. Comitar a resolução
git commit -m "resolve: merge conflicts in detection module"

# 5. Fazer push
git push origin main
```

## 7. Comandos de Emergência

### Desfazer último commit (se não fez push ainda)
```bash
git reset HEAD~1  # Mantém mudanças nos arquivos
git reset --hard HEAD~1  # Remove mudanças completamente
```

### Desfazer mudanças em arquivo
```bash
git checkout -- arquivo.py  # Desfaz mudanças não commitadas
```

### Ver histórico detalhado
```bash
git log --graph --oneline --all  # Visualização gráfica
git show commit_hash  # Ver detalhes de commit específico
```

## 8. Estado Atual do Projeto

✅ **Repositório limpo e atualizado**
✅ **Documentação completa criada**
✅ **Scripts de automação implementados**
✅ **Estrutura organizada**

**Próximos passos sugeridos:**
1. Implementar testes automatizados
2. Adicionar CI/CD pipeline
3. Melhorar documentação técnica
4. Otimizar performance do modelo

---

**Contato para dúvidas**: Use Issues no GitHub ou comunicação interna da equipe
