# 🚀 Guia Rápido: Fine-Tuning no Google Colab

## ✅ O que foi otimizado

### 1. **Datasets Limpos** (Economiza ~7% de tokens)
- ✓ `train_llama3_optimized.json` - 582 exemplos (2.6 MB)
- ✓ `test_llama3_optimized.json` - 146 exemplos (551 KB)
- Removidos espaços excessivos, quebras de linha múltiplas, tabs

### 2. **Notebook Otimizado**
- ✓ **Unsloth** - Biblioteca especializada para Colab (2x mais rápida)
- ✓ **Meta-Llama-3.1-70B-Instruct** com quantização 4-bit
- ✓ **LoRA** - Treina apenas ~0.5% dos parâmetros
- ✓ **Validação automática** durante treinamento

---

## 📋 Passo a Passo

### **1. Preparar arquivos**

Faça upload para seu Google Drive:
```
MyDrive/
└── TechChallenge/
    ├── train_llama3_optimized.json  ← Upload deste arquivo
    └── test_llama3_optimized.json   ← Upload deste arquivo
```

**Localização dos arquivos:**
- `data/finetuning/train_llama3_optimized.json`
- `data/finetuning/test_llama3_optimized.json`

---

### **2. Abrir Google Colab**

1. Acesse: https://colab.research.google.com
2. File → Upload notebook
3. Upload: `notebooks/Llama3_Medical_FineTuning_Colab.ipynb`

---

### **3. Configurar GPU**

⚠️ **IMPORTANTE:** Sem GPU não funciona!

1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU** (ou L4 se disponível)
3. Save

Verifique:
```python
# Célula 3 mostrará:
# ✓ GPU: Tesla T4
# ✓ Memória GPU: 15.00 GB
```

---

### **4. Ajustar paths**

Na **Célula 4**, ajuste o caminho do seu Drive:

```python
# ANTES (padrão):
BASE_PATH = "/content/drive/MyDrive/TechChallenge"

# AJUSTE para o SEU caminho:
BASE_PATH = "/content/drive/MyDrive/SEU_CAMINHO_AQUI"
```

---

### **5. Executar notebook**

**Opção 1:** Executar tudo
- Runtime → Run all (Ctrl+F9)

**Opção 2:** Célula por célula
- Cell → Run cell (Ctrl+Enter)
- Aguarde cada célula terminar antes da próxima

---

## ⏱️ Tempo Esperado

| Etapa | Tempo |
|-------|-------|
| Instalação de dependências | ~3-5 min |
| Carregar modelo 70B | ~3-5 min |
| Treinamento (3 epochs) | ~30-45 min |
| Salvar modelo | ~2 min |
| **TOTAL** | **~40-60 min** |

---

## 📊 Monitoramento

Durante o treinamento (Célula 11), você verá:

```
Step   | Training Loss | Validation Loss | Time
-------|---------------|-----------------|------
10     | 2.4567       | 2.3456         | 0:02
20     | 2.1234       | 2.0987         | 0:05
...    | ...          | ...            | ...
```

✅ **Bom sinal:** Loss diminuindo progressivamente
⚠️ **Atenção:** Se loss parar de diminuir, pode ser overfitting

---

## 💾 Download do Modelo Treinado

Após conclusão (Célula 13):

1. No Google Drive, vá para: `TechChallenge/llama3_medical_ft/`
2. Clique com botão direito → Download
3. Extraia a pasta completa
4. Cole em: `models/llama3_medical_ft/`

**Arquivos esperados:**
```
models/llama3_medical_ft/
├── adapter_config.json      (~2 KB)
├── adapter_model.safetensors (~45 MB)
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── ...
```

---

## 🧪 Testar Localmente

Após download, teste com:

```powershell
python scripts/test_modelo_treinado.py
```

---

## ❌ Troubleshooting

### **Erro: GPU não detectada**
```
Solução: Runtime → Change runtime type → T4 GPU
```

### **Erro: Arquivos não encontrados**
```
Solução: Verifique paths na Célula 4
         Confirme upload dos JSONs no Drive
```

### **Erro: Out of Memory**
```
Solução: 
- Reduza batch_size de 2 para 1 (Célula 10)
- Reduza max_seq_length de 2048 para 1024 (Célula 5)
```

### **Erro: Unsloth não instalado**
```
Solução: Execute Célula 2 novamente
         Restart Runtime se necessário
```

### **Treinamento muito lento (>2h)**
```
Possíveis causas:
- GPU não ativada (usando CPU)
- Batch size muito pequeno
- Dataset muito grande (use versão small para testes)
```

---

## 🔍 Diferenças vs Notebook Anterior

| Aspecto | Anterior | Novo (Unsloth) |
|---------|----------|----------------|
| Biblioteca | transformers + peft | Unsloth (otimizado) |
| Velocidade | Normal | 2x mais rápido |
| Memória | Alta | 30% menos memória |
| Compatibilidade | Vários problemas | Testado para Colab |
| Validação | Manual | Automática |
| Template | Manual | Automático (Llama-3) |

---

## 📈 Métricas de Sucesso

✅ **Treinamento bem-sucedido se:**
- Training loss < 1.5
- Validation loss < 2.0
- Modelo responde coerentemente (Célula 12)
- Arquivos salvos corretamente

---

## 💡 Dicas

1. **Primeira vez:** Execute célula por célula para entender o processo
2. **Economize créditos:** Feche sessão Colab após download
3. **Backup:** Mantenha modelo no Drive (não confie apenas no local)
4. **Experimente:** Ajuste parâmetros na Célula 10 para melhorar resultados

---

## 📚 Próximos Passos

Após fine-tuning bem-sucedido:

1. ✅ Testar localmente (`test_modelo_treinado.py`)
2. ✅ Integrar com RAG (`src/agents/medical_agent.py`)
3. ✅ Benchmark vs modelo base
4. ✅ Documentar resultados para Tech Challenge

---

## 🎯 Tech Challenge - Checklist

- [ ] Fine-tuning concluído no Colab
- [ ] Modelo baixado e extraído em `models/`
- [ ] Testes locais bem-sucedidos
- [ ] Integração com RAG funcionando
- [ ] Benchmark documentado
- [ ] Notebook de demonstração criado

---

**Qualquer dúvida, verifique os logs de cada célula!** 🚀
