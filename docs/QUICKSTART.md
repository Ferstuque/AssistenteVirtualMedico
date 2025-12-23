# 🚀 Guia de Início Rápido

## Configuração em 5 Minutos

### 1️⃣ Clone e Instale

```bash
# Clone o repositório
git clone <repository-url>
cd AssistenteVirtualMedico

# Crie ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instale dependências
pip install -r requirements.txt

# Instale PyTorch (escolha a versão apropriada)
# GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Ou CPU-only:
pip install torch torchvision torchaudio
```

### 2️⃣ Configure API Keys

Crie arquivo `.env` na raiz do projeto:

```bash
# Copie o template
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edite e adicione suas chaves
notepad .env  # Windows
nano .env     # Linux/Mac
```

Conteúdo do `.env`:

```env
# Google Gemini API Key
GOOGLE_API_KEY=sua_chave_google_aqui

# Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN=seu_token_huggingface_aqui
```

**Como obter as chaves:**
- Google Gemini: https://makersuite.google.com/app/apikey
- Hugging Face: https://huggingface.co/settings/tokens

### 3️⃣ Execute o Sistema

#### Opção A: Interface Interativa (Recomendado)

```bash
python src/main.py
```

O sistema iniciará uma interface de chat interativa no terminal.

#### Opção B: Notebook Jupyter (Demonstração Completa)

```bash
# Inicie o Jupyter
jupyter notebook

# Abra o notebook:
# notebooks/demonstração_completa_tech_challenge.ipynb
```

O notebook demonstra:
- ✅ Análise de prontuários de pacientes
- ✅ Consultas médicas gerais
- ✅ Sistema RAG com explicabilidade
- ✅ Avaliação de métricas

---

## 📝 Exemplos de Uso

### Consulta Médica Geral

```
Você: Quais são os principais sintomas de câncer de pulmão?

Assistente: [Resposta detalhada com informações do RAG + fontes]
```

### Análise de Prontuário

```
Você: Analise o prontuário do paciente João Silva

Assistente: [Análise completa com prognóstico e recomendações]
```

### Perguntas Específicas sobre Paciente

```
Você: Qual o histórico médico do paciente com ID brcp230442?

Assistente: [Histórico detalhado e análise de riscos]
```

---

## 🎯 Funcionalidades Principais

### 1. Sistema RAG Inteligente
- Recuperação de contexto médico confiável
- Embeddings com FAISS
- Top-5 chunks mais relevantes

### 2. Multi-Agente com LangGraph
- **Llama 3.1 70B**: Raciocínio clínico complexo (fine-tuned)
- **Gemini 1.5**: Respostas gerais + RAG
- Orquestração inteligente entre modelos

### 3. Guardrails de Segurança
- Validação ética de perguntas
- Detecção de alucinações
- Confiabilidade das respostas

### 4. Análise de Prontuários
- 25 prontuários fictícios oncológicos
- Busca por ID ou nome
- Prognóstico baseado em histórico

### 5. Explainability Completa
- Scores de relevância para cada chunk
- Fontes citadas
- Métricas de confiabilidade

---

## 🏗️ Estrutura do Projeto

```
AssistenteVirtualMedico/
├── src/                    # Código-fonte principal
│   ├── agents/            # Agentes LLM (Llama, Gemini)
│   ├── rag/               # Pipeline RAG
│   ├── guardrails/        # Validações de segurança
│   ├── finetuning/        # Scripts de fine-tuning
│   └── utils/             # Utilitários (logger, metrics)
├── data/                   # Dados do sistema
│   ├── raw/               # XMLs do MedQuAD
│   ├── processed/         # Prontuários JSON
│   └── vectorstore/       # FAISS vector store
├── docs/                   # Documentação técnica
├── notebooks/             # Notebooks de demonstração
├── scripts/               # Scripts auxiliares
└── config/                # Configurações
```

---

## 📊 Verificar Instalação

Execute os testes para verificar se tudo está funcionando:

```bash
# Verificar configuração
python -c "from config.config import get_settings; print('✅ Config OK')"

# Verificar RAG
python -m pytest tests/test_rag.py

# Verificar guardrails
python -m pytest tests/test_guardrails.py

# Verificar métricas
python -m pytest tests/test_metrics.py
```

---

## 🐛 Solução de Problemas

### Erro: "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### Erro: "GPU não detectada"
Instale PyTorch com CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Erro: "API Key inválida"
Verifique se o arquivo `.env` está na raiz e contém as chaves corretas.

### Erro: "FAISS index not found"
Execute o notebook ou main.py uma vez para criar o vector store.

---

## 📚 Próximos Passos

1. **Explore o Notebook**: [notebooks/demonstração_completa_tech_challenge.ipynb](../notebooks/demonstração_completa_tech_challenge.ipynb)
2. **Leia a Arquitetura**: [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Veja o Fluxo Completo**: [FLUXO_SISTEMA.md](FLUXO_SISTEMA.md)
4. **Análise de Prontuários**: [ANALISE_PRONTUARIOS.md](ANALISE_PRONTUARIOS.md)
5. **Fine-Tuning**: [GUIA_COLAB_FINETUNING.md](GUIA_COLAB_FINETUNING.md)

---

## 🤝 Suporte

Para problemas ou dúvidas:
- Verifique a documentação em [docs/](../docs/)
- Consulte o README principal
- Revise os logs em `logs/`

---

## ✅ Checklist de Configuração

- [ ] Python 3.12+ instalado
- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`requirements.txt`)
- [ ] PyTorch instalado (com ou sem CUDA)
- [ ] Arquivo `.env` criado com API keys
- [ ] Sistema executado com sucesso
- [ ] Notebook explorado

🎉 Pronto! Seu Assistente Virtual Médico está configurado!
