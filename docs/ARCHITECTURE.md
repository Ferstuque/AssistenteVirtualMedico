# 🔧 Guia de Arquitetura e Decisões Técnicas

## 📐 Arquitetura do Sistema

### Visão Geral

O sistema foi projetado seguindo princípios de:
- **Modularidade**: Cada componente é independente e reutilizável
- **Escalabilidade**: Arquitetura permite adicionar novos agentes facilmente
- **Observabilidade**: Logging e métricas em todos os níveis
- **Segurança**: Guardrails em múltiplas camadas

### Componentes Principais

#### 1. Pipeline RAG (Retrieval-Augmented Generation)

**Localização**: `src/rag/pipeline.py`

**Responsabilidades**:
- Chunking de documentos médicos
- Geração de embeddings
- Armazenamento em vector store (FAISS)
- Retrieval contextual

**Decisões Técnicas**:
- **FAISS vs ChromaDB**: Escolhemos FAISS por:
  - Melhor performance em CPU
  - Menor overhead de memória
  - Funciona offline após construção
  
- **Chunking Strategy**: 
  - Tamanho: 500 caracteres
  - Overlap: 50 caracteres
  - Separadores hierárquicos para manter contexto

**Trade-offs**:
- ✅ Rápido e eficiente
- ❌ Requer rebuild completo para novos dados

#### 2. Agente Llama (Raciocínio Clínico)

**Localização**: `src/agents/llama_agent.py`

**Responsabilidades**:
- Análise de casos clínicos
- Raciocínio diferencial
- Decisões de urgência
- Recomendações estruturadas

**Decisões Técnicas**:
- **Modelo**: Llama-2-7b-chat
  - Tamanho equilibrado (7B parâmetros)
  - Especializado em conversação
  - Open-source
  
- **Temperatura**: 0.3
  - Baixa para respostas mais determinísticas
  - Apropriado para contexto médico

**Trade-offs**:
- ✅ Excelente em raciocínio estruturado
- ❌ Resposta em inglês (requer pós-processamento)
- ❌ Requer HuggingFace API

#### 3. Agente Gemini (RAG & Insights)

**Localização**: `src/agents/gemini_agent.py`

**Responsabilidades**:
- Geração de respostas em PT-BR
- Integração com RAG
- Citação de fontes
- Insights contextuais

**Decisões Técnicas**:
- **Modelo**: Gemini Pro
  - Excelente em português
  - Boa integração com contexto
  - API estável do Google
  
- **Temperatura**: 0.7
  - Balanceada entre criatividade e precisão
  - Permite respostas naturais

**Trade-offs**:
- ✅ Melhor performance em PT-BR
- ✅ Respostas mais naturais
- ❌ Dependência de API externa
- ❌ Rate limits a considerar

#### 4. Orquestrador (LangGraph)

**Localização**: `src/orchestrator.py`

**Responsabilidades**:
- Coordenação entre agentes
- Roteamento inteligente
- Gerenciamento de estado
- Validação de fluxo

**Decisões Técnicas**:
- **LangGraph vs Chains Simples**:
  - ✅ Controle fino sobre fluxo
  - ✅ Estado compartilhado
  - ✅ Decisões condicionais
  - ✅ Fácil debugging

**Fluxo de Execução**:

```
1. Classificação
   ├─ Clínico? → Llama + Gemini
   └─ Informativo? → Apenas Gemini

2. Retrieval (sempre)
   └─ Busca contextual no vector store

3. Processamento
   ├─ Llama: Raciocínio clínico
   └─ Gemini: Resposta final

4. Validação (Guardrails)
   ├─ Passa? → Retorna
   └─ Falha? → Mensagem segura

5. Logging
   └─ Registra tudo
```

#### 5. Guardrails (Pydantic)

**Localização**: `src/guardrails/validators.py`

**Responsabilidades**:
- Validação de respostas
- Bloqueio de prescrições
- Verificação de PII
- Garantia de qualidade

**Decisões Técnicas**:
- **Pydantic v2**: 
  - Validação em runtime
  - Type hints nativos
  - Performance otimizada
  
**Regras Implementadas**:

1. **Anti-Prescrição**:
   ```python
   # Bloqueia:
   - "tome X mg"
   - "prescrevo"
   - "dose de X"
   ```

2. **Anti-PII**:
   ```python
   # Bloqueia:
   - Nomes próprios completos
   - CPF/documentos
   - Emails/telefones
   ```

3. **Linguagem Profissional**:
   ```python
   # Bloqueia:
   - "cura definitiva"
   - "garantido"
   - "milagroso"
   ```

#### 6. Sistema de Logging

**Localização**: `src/utils/logger.py`

**Responsabilidades**:
- Logging estruturado
- Rastreamento de decisões
- Métricas de performance
- Auditoria

**Decisões Técnicas**:
- **Structlog**:
  - JSON estruturado
  - Contexto rico
  - Fácil parsing
  
**Eventos Capturados**:
- Requisições de usuário
- Decisões de agentes
- Retrieval do RAG
- Violações de guardrails
- Erros e exceções
- Métricas de tempo

#### 7. Sistema de Métricas

**Localização**: `src/utils/metrics.py`

**Responsabilidades**:
- Avaliação de qualidade
- BLEU, ROUGE, F1
- Relatórios de performance

**Métricas Implementadas**:

1. **BLEU (1-4 gramas)**:
   - Avalia n-gramas em comum
   - Score: 0-1
   
2. **ROUGE (1, 2, L)**:
   - Precision, Recall, F1
   - Overlap de palavras/frases

3. **Classificação**:
   - Precision, Recall, F1
   - Accuracy

4. **Perplexidade**:
   - Qualidade do LM
   - Menor = melhor

---

## 🔐 Segurança e Compliance

### Anonimização

**Biblioteca**: Microsoft Presidio

**Estratégia**:
1. Análise de texto para PII
2. Reconhecimento de entidades
3. Substituição por tokens
4. Validação final

**Entidades Detectadas**:
- PERSON (nomes)
- EMAIL_ADDRESS
- PHONE_NUMBER
- LOCATION
- DATE_TIME
- PATIENT_ID
- MEDICAL_RECORD

### Guardrails em Múltiplas Camadas

**Camada 1**: Input Validation
- Valida query do usuário
- Remove conteúdo malicioso

**Camada 2**: Processing Validation
- Valida contexto recuperado
- Verifica sources

**Camada 3**: Output Validation
- Guardrails Pydantic
- Bloqueio de prescrições
- Verificação de PII

**Camada 4**: Post-Processing
- Logging de violações
- Métricas de segurança

---

## 📊 Monitoramento e Observabilidade

### Logs Estruturados

Formato JSON para fácil parsing:

```json
{
  "event": "agent_decision",
  "agent": "gemini",
  "decision": "...",
  "confidence": 0.85,
  "sources": ["chunk_001", "chunk_002"],
  "timestamp": "2025-12-13T10:30:00",
  "level": "info"
}
```

### Métricas de Performance

Capturadas automaticamente:
- Tempo de processamento por agente
- Número de chunks recuperados
- Scores de similaridade
- Taxa de sucesso de guardrails

### Rastreamento de Decisões

Cada resposta inclui:
- Passos executados
- Agentes utilizados
- Fontes consultadas
- Tempo em cada etapa

---

## 🚀 Otimizações Futuras

### Performance

1. **Caching**:
   - Cache de embeddings
   - Cache de respostas comuns
   
2. **Batching**:
   - Processar múltiplas queries
   - Embeddings em batch

3. **Quantização**:
   - Modelos quantizados (4-bit)
   - Menor uso de memória

### Funcionalidades

1. **Fine-tuning**:
   - Treinar Llama em dados médicos PT-BR
   - Adapter layers (LoRA)
   
2. **Multi-modal**:
   - Processar imagens médicas
   - Análise de exames

3. **Feedback Loop**:
   - Aprender com correções
   - Melhorar continuamente

### Escalabilidade

1. **Distribuição**:
   - Kubernetes deployment
   - Load balancing
   
2. **Banco de Dados**:
   - PostgreSQL para histórico
   - Redis para cache

3. **APIs**:
   - REST API
   - WebSocket para streaming

---

## 📝 Lições Aprendidas

### O que funcionou bem

✅ **Arquitetura Multi-Agente**:
- Separação de responsabilidades clara
- Fácil manutenção e extensão

✅ **RAG com FAISS**:
- Performance excelente
- Setup simples

✅ **Guardrails Pydantic**:
- Type-safe
- Validação robusta

### Desafios Enfrentados

⚠️ **Latência de APIs**:
- Solução: Processamento assíncrono
- Melhoria: Cache inteligente

⚠️ **Qualidade das Respostas**:
- Solução: Prompts iterativos
- Melhoria: Fine-tuning específico

⚠️ **Custos de API**:
- Solução: Rate limiting
- Melhoria: Modelos locais para dev

---

## 🔍 Referências Técnicas

### Papers

- **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **LangGraph**: LangChain Documentation
- **Llama 2**: Meta AI Research Paper

### Bibliotecas

- LangChain: https://python.langchain.com
- Presidio: https://microsoft.github.io/presidio
- FAISS: https://github.com/facebookresearch/faiss

### Datasets

- MedQuAD: https://github.com/abachaa/MedQuAD
- Medical Q&A Datasets

---

**Última atualização**: Dezembro 2025
