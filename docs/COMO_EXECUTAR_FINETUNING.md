# 🚀 Como Executar o Fine-Tuning do Llama3

Este guia explica como treinar o modelo Llama-3.2-3B-Instruct com os datasets médicos limpos e validados.

## 📋 Pré-requisitos

### Hardware Recomendado

| Configuração | GPU | RAM | VRAM | Tempo Estimado |
|-------------|-----|-----|------|----------------|
| **Ideal** | NVIDIA RTX 3090/4090 | 32GB | 24GB | ~2-3 horas |
| **Mínimo GPU** | NVIDIA GTX 1080 Ti | 16GB | 11GB | ~4-6 horas |
| **CPU (lento)** | - | 32GB+ | - | ~24-48 horas |

### Software

```bash
# Python 3.10+
python --version

# CUDA (se usar GPU)
nvidia-smi

# Pacotes Python necessários
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install trl>=0.8.0
pip install bitsandbytes>=0.43.0
pip install accelerate>=0.28.0
pip install evaluate>=0.4.1
pip install rouge-score
pip install sacrebleu
pip install tensorboard
```

## 🎯 Passo a Passo

### 1. Verificar Datasets

Certifique-se que os datasets limpos existem:

```bash
ls -lh data/finetuning/

# Deve mostrar:
# train_llama3_optimized.json (2.8 MB, 582 exemplos)
# test_llama3_optimized.json (589 KB, 146 exemplos)
# dataset_metadata.json
```

### 2. Executar Fine-Tuning (RECOMENDADO: Fora do Notebook)

#### Configuração Padrão

```bash
# Ativar ambiente virtual
.\venv\Scripts\activate

# Executar treinamento
python scripts/finetune_llama3.py
```

**Configurações padrão:**
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-4
- Gradient accumulation: 4 (batch efetivo = 16)
- Quantização: 4-bit (QLoRA)
- LoRA rank: 16

#### Configuração Customizada

```bash
# Treinar por mais epochs
python scripts/finetune_llama3.py --epochs 5

# Batch size maior (requer mais VRAM)
python scripts/finetune_llama3.py --batch_size 8

# Learning rate menor
python scripts/finetune_llama3.py --learning_rate 1e-4

# Combinação
python scripts/finetune_llama3.py --epochs 5 --batch_size 8 --learning_rate 1e-4
```

### 3. Monitorar Treinamento

#### Logs em Tempo Real

```bash
# Terminal 1: Executar treinamento
python scripts/finetune_llama3.py

# Terminal 2: Acompanhar logs
tail -f logs/finetuning.log
```

#### TensorBoard (Métricas Visuais)

```bash
# Iniciar TensorBoard
tensorboard --logdir logs/tensorboard

# Acessar no navegador
http://localhost:6006
```

### 4. Avaliar Modelo Treinado

```bash
# Avaliação completa no test set
python scripts/evaluate_model.py

# Modo interativo (testar perguntas manualmente)
python scripts/evaluate_model.py --interactive
```

---

### Opção 2: Direto do Notebook

**Vantagens:**
- ✅ Tudo em um lugar
- ✅ Vê output no notebook

**Desvantagens:**
- ❌ Bloqueia o notebook por horas
- ❌ Se fechar VS Code, perde o progresso

**Como fazer:**

1. Vá até a **célula 18** do notebook
2. Remova as aspas triplas (`"""`)
3. Execute a célula
4. Aguarde 6-12 horas

---

### Opção 3: Google Colab (Cloud)

**Vantagens:**
- ✅ GPU gratuita (T4) ou paga (A100)
- ✅ Mais rápido
- ✅ Não usa recursos locais

**Como fazer:**

1. **Upload dos dados:**
   - `data/finetuning/train_llama3_optimized.json`
   - `data/finetuning/test_llama3_optimized.json`
   - `data/finetuning/config.json`
   - `src/finetuning/train_llama.py`

2. **No Colab, criar célula:**
   ```python
   !pip install transformers peft bitsandbytes accelerate datasets
   !python train_llama.py \
       --train_data train_llama3.json \
       --test_data test_llama3.json \
       --config config.json \
       --output_dir llama3_medical_ft
   ```

3. **Baixar modelo treinado:**
   - Fazer download da pasta `llama3_medical_ft/final/`
   - Colocar em `models/llama3_medical_ft/final/` local

---

## 📊 O Que Esperar Durante Fine-Tuning

### Início
```
🖥️ Dispositivo: cuda:0 (NVIDIA RTX 4090)
📦 Carregando modelo: meta-llama/Meta-Llama-3-8B-Instruct
📊 Parâmetros treináveis: 8,388,608 / 8,030,261,248 (1.04%)
📚 Dataset preparado: 583 exemplos (treino)
🏋️ Treinando modelo...
```

### Durante (cada 10 steps)
```
Step 10/1749 | Loss: 2.345 | LR: 0.000198 | 15.2 steps/sec
Step 20/1749 | Loss: 2.123 | LR: 0.000196 | 15.1 steps/sec
...
```

### Checkpoints (cada 500 steps)
```
💾 Salvando checkpoint: checkpoint-500
✅ Checkpoint salvo!
```

### Final
```
✅ Fine-tuning concluído!
📂 Modelo salvo em: models/llama3_medical_ft
⏱️ Tempo total: 8h 23min
```

---

## ⏱️ Tempo de Treinamento

| GPU | Batch Size | Tempo Estimado |
|-----|------------|----------------|
| A100 (80GB) | 8 | 4-6 horas |
| RTX 4090 (24GB) | 4 | 6-8 horas |
| RTX 3090 (24GB) | 2 | 8-12 horas |
| T4 (16GB Colab) | 1 | 12-16 horas |
| CPU | N/A | ⚠️ Dias/Semanas |

---

## 🔍 Monitorar Progresso

### Opção 1: Terminal
- Veja output em tempo real
- `Ctrl+C` para parar (salva checkpoint atual)

### Opção 2: nvidia-smi (outra janela)
```bash
# Monitorar uso de GPU
nvidia-smi -l 1
```

### Opção 3: Logs
```bash
# Ver logs salvos
tail -f models/llama3_medical_ft/training.log
```

---

## 🛑 Parar e Retomar

### Parar
- Terminal: `Ctrl+C`
- Notebook: Interrupt kernel
- Último checkpoint é salvo automaticamente

### Retomar
```bash
python src/finetuning/train_llama.py \
    --train_data data/finetuning/train_llama3.json \
    --test_data data/finetuning/test_llama3.json \
    --config data/finetuning/config.json \
    --output_dir models/llama3_medical_ft \
    --resume_from_checkpoint models/llama3_medical_ft/checkpoint-500
```

---

## ✅ Verificar se Funcionou

Após fine-tuning, teste o modelo:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Carregar modelo treinado
model_path = "models/llama3_medical_ft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Testar
prompt = "O que é leucemia?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0])

print(response)
```

---

## 🚨 Troubleshooting

### "CUDA out of memory"
**Solução:**
1. Reduzir `batch_size` em `config.json`
2. Aumentar `gradient_accumulation_steps`
3. Reduzir `max_seq_length`

### "Training very slow"
**Verificar:**
1. GPU está sendo usada? (`nvidia-smi`)
2. FP16 ativado? (verifica `config.json`)
3. Drivers NVIDIA atualizados?

### "Process killed"
**Causa:** RAM insuficiente

**Solução:**
1. Fechar outros programas
2. Reduzir `batch_size`
3. Usar `gradient_checkpointing`

---

## 💡 Dicas

1. **Faça backup dos checkpoints** - Copie periodicamente
2. **Use terminal** - Não bloqueia outras tarefas
3. **Monitore temperatura da GPU** - Use ferramentas de monitoramento
4. **Teste em cloud primeiro** - Google Colab gratuito para testar
5. **Documente configurações** - Anote o que funcionou

---

## 📚 Próximos Passos

Após fine-tuning:

1. ✅ Modelo salvo em `models/llama3_medical_ft`
2. ✅ Atualizar `llama_agent.py` para usar modelo treinado
3. ✅ Testar com perguntas médicas
4. ✅ Comparar com modelo base
5. ✅ Avaliar métricas (ROUGE, BLEU)

---

## 🆘 Precisa de Ajuda?

- [FINETUNING_GPU.md](docs/FINETUNING_GPU.md) - Configurações GPU
- [SOLUCAO_GPU.md](docs/SOLUCAO_GPU.md) - Troubleshooting
- Execute células de diagnóstico do notebook
