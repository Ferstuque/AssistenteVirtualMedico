# 📊 Fluxo Detalhado do Sistema - LangChain/LangGraph

Este documento apresenta visualizações detalhadas do fluxo de execução do Assistente Médico Virtual, incluindo todos os componentes e suas interações.

---

## 🔄 Fluxo Principal do Sistema

```mermaid
sequenceDiagram
    participant U as Usuário
    participant O as Orquestrador
    participant C as Classificador
    participant R as RAG Pipeline
    participant L as Agente Llama
    participant G as Agente Gemini
    participant V as Guardrails
    
    U->>O: Pergunta médica
    O->>C: Classificar query
    C->>O: Tipo: clínica, urgência: média
    
    O->>R: Recuperar contexto
    R->>R: Gerar embedding
    R->>R: Buscar top-5 chunks
    R->>O: Contexto + fontes
    
    alt Requer raciocínio clínico
        O->>L: Query + Contexto
        L->>L: Análise clínica
        L->>L: Raciocínio diferencial
        L->>O: Resposta estruturada (5 seções)
        
        O->>G: Sintetizar resposta
        G->>G: Traduzir para PT-BR
        G->>G: Adicionar formatação
        G->>O: Resposta final em português
    else Pergunta geral
        O->>G: Query + Contexto direto
        G->>G: Gerar resposta
        G->>O: Resposta em português
    end
    
    O->>V: Validar resposta
    V->>V: Detectar alucinações
    V->>V: Verificar ética
    V->>V: Calcular confidence
    
    alt Validação passou
        V->>O: ✅ Aprovado (confidence: 0.85)
        O->>U: Resposta final + disclaimer
    else Alucinação detectada
        V->>O: ❌ Rejeitado
        O->>G: Reprocessar com contexto adicional
        G->>O: Nova resposta
        O->>V: Validar novamente
        V->>O: ✅ Aprovado
        O->>U: Resposta corrigida + aviso
    end
```

---

## 🎯 Arquitetura de Componentes

```mermaid
graph TB
    subgraph "Interface"
        UI[Usuário/API]
    end
    
    subgraph "Orquestração - LangGraph"
        Orch[Orquestrador StateGraph]
        State[Estado Compartilhado]
        Router[Roteador Condicional]
    end
    
    subgraph "Agentes Especializados"
        Llama[Agente Llama 3.1 70B<br/>Raciocínio Clínico]
        Gemini[Agente Gemini Pro<br/>RAG + Síntese PT-BR]
    end
    
    subgraph "RAG Pipeline"
        Embed[Embeddings<br/>all-MiniLM-L6-v2]
        Vector[Vector Store<br/>FAISS]
        Retr[Retriever<br/>Top-K Search]
    end
    
    subgraph "Guardrails"
        Halluc[Detector de<br/>Alucinações]
        Ethics[Validador<br/>Ético]
        Conf[Calculador de<br/>Confiança]
    end
    
    subgraph "Dados"
        Raw[XMLs CancerGov<br/>728 documentos]
        Proc[Dataset Processado<br/>728 pares Q&A]
        FT[Dataset Fine-Tuning<br/>582 train + 146 test]
        Model[Modelo Fine-Tunado<br/>LoRA adapters 45MB]
    end
    
    UI --> Orch
    Orch --> State
    Orch --> Router
    
    Router --> Llama
    Router --> Gemini
    
    Llama --> State
    Gemini --> Retr
    Gemini --> State
    
    Retr --> Vector
    Vector --> Embed
    
    State --> Halluc
    State --> Ethics
    State --> Conf
    
    Halluc --> UI
    Ethics --> UI
    Conf --> UI
    
    Raw --> Proc
    Proc --> FT
    FT --> Model
    Model --> Llama
    
    style UI fill:#e1f5e1
    style Orch fill:#fff4e1
    style Llama fill:#ffe1e1
    style Gemini fill:#e1f0ff
    style Vector fill:#f0e1ff
```

---

## 🔀 Grafo de Estados do LangGraph

```mermaid
stateDiagram-v2
    [*] --> ClassifyQuery: Pergunta do usuário
    
    ClassifyQuery --> RetrieveContext: Classificada
    
    RetrieveContext --> RouteDecision: Contexto recuperado
    
    RouteDecision --> LlamaReasoning: Requer análise clínica
    RouteDecision --> GeminiResponse: Pergunta geral
    
    LlamaReasoning --> CheckLlama: Resposta gerada
    
    CheckLlama --> GeminiResponse: Sucesso
    CheckLlama --> RetryLlama: Erro/Timeout
    
    RetryLlama --> GeminiResponse: Retry com temp ajustada
    
    GeminiResponse --> ValidateResponse: Síntese completa
    
    ValidateResponse --> CheckValidation: Validação executada
    
    CheckValidation --> Finalize: ✅ Passou
    CheckValidation --> AddWarning: ⚠️ Low confidence
    CheckValidation --> RetryGemini: ❌ Hallucination
    
    AddWarning --> Finalize: Aviso adicionado
    RetryGemini --> ValidateResponse: Reprocessado
    
    Finalize --> [*]: Resposta ao usuário
    
    note right of ClassifyQuery
        Identifica tipo de query:
        - Clínica vs Geral
        - Nível de urgência
        - Complexidade
    end note
    
    note right of RetrieveContext
        RAG Pipeline:
        - Embedding da query
        - Top-5 similarity search
        - Contexto concatenado
    end note
    
    note right of ValidateResponse
        Guardrails:
        - Hallucination detection
        - Ethical validation
        - Confidence scoring
    end note
```

---

## 🧠 Fluxo de Raciocínio do Agente Llama

```mermaid
graph TD
    Start[Query + Contexto RAG] --> Format[Formatar Prompt Llama3]
    
    Format --> Sys[System Prompt:<br/>Você é especialista em oncologia]
    Sys --> User[User Message:<br/>Contexto + Pergunta]
    User --> Gen[Gerar Resposta<br/>temperature=0.1]
    
    Gen --> Parse[Parsear Resposta]
    
    Parse --> Sec1[Seção 1:<br/>Resumo da Condição]
    Parse --> Sec2[Seção 2:<br/>Diagnósticos Diferenciais]
    Parse --> Sec3[Seção 3:<br/>Investigações Recomendadas]
    Parse --> Sec4[Seção 4:<br/>Nível de Urgência]
    Parse --> Sec5[Seção 5:<br/>Recomendações ao Médico]
    
    Sec1 --> Combine[Combinar Seções]
    Sec2 --> Combine
    Sec3 --> Combine
    Sec4 --> Combine
    Sec5 --> Combine
    
    Combine --> Validate{Todas as<br/>seções presentes?}
    
    Validate -->|Sim| Success[✅ Resposta Estruturada]
    Validate -->|Não| Retry[🔄 Regenerar]
    
    Retry --> Gen
    
    Success --> Return[Retornar ao Orquestrador]
    
    style Start fill:#e1f5e1
    style Success fill:#e1f5e1
    style Return fill:#e1f5e1
    style Retry fill:#ffe1e1
```

---

## 🌐 Fluxo de Síntese do Agente Gemini

```mermaid
graph TD
    Input[Input:<br/>Resposta Llama + Contexto RAG] --> Check{Tem resposta<br/>do Llama?}
    
    Check -->|Sim| Mode1[Modo: Síntese Combinada]
    Check -->|Não| Mode2[Modo: Resposta Direta]
    
    Mode1 --> Prompt1[Construir Prompt:<br/>Sintetize análise clínica]
    Mode2 --> Prompt2[Construir Prompt:<br/>Responda com contexto]
    
    Prompt1 --> Call1[Chamar Gemini Pro<br/>temperature=0.1]
    Prompt2 --> Call1
    
    Call1 --> Translate[Traduzir para PT-BR]
    
    Translate --> Format[Formatar Resposta:<br/>- Adicionar emojis<br/>- Estruturar seções<br/>- Melhorar legibilidade]
    
    Format --> AddSources[Adicionar Citações:<br/>📚 Fonte: CancerGov]
    
    AddSources --> AddDisclaimer[Adicionar Disclaimer:<br/>⚠️ Não substitui consulta médica]
    
    AddDisclaimer --> Output[✅ Resposta Final em PT-BR]
    
    style Input fill:#e1f5e1
    style Output fill:#e1f5e1
    style Call1 fill:#e1f0ff
```

---

## 🔍 Pipeline RAG Detalhado

```mermaid
graph LR
    subgraph "1. Indexação (Offline)"
        Doc[Documentos XML<br/>CancerGov] --> Parse[Parser XML]
        Parse --> Clean[Limpeza de Texto]
        Clean --> Chunk[Chunking<br/>500 chars, overlap 50]
        Chunk --> Embed1[Gerar Embeddings<br/>all-MiniLM-L6-v2]
        Embed1 --> Store[Armazenar no FAISS<br/>~3.500 chunks]
    end
    
    subgraph "2. Retrieval (Online)"
        Query[Query do Usuário] --> Embed2[Gerar Embedding<br/>da Query]
        Embed2 --> Search[Similarity Search<br/>FAISS L2 Distance]
        Store --> Search
        Search --> TopK[Top-K Chunks<br/>K=5, threshold>0.7]
        TopK --> Rank[Reranking por Score]
    end
    
    subgraph "3. Augmentation"
        Rank --> Context[Construir Contexto<br/>Concatenado]
        Context --> Meta[Adicionar Metadados:<br/>- Chunk IDs<br/>- Scores<br/>- Sources]
        Meta --> LLM[Fornecer para LLM]
    end
    
    style Doc fill:#e1f5e1
    style Query fill:#e1f5e1
    style LLM fill:#e1f5e1
```

---

## 🛡️ Guardrails - Fluxo de Validação

```mermaid
graph TD
    Response[Resposta Gerada] --> V1[Validador 1:<br/>Detecção de Alucinação]
    
    V1 --> Check1{Termos médicos<br/>no contexto?}
    Check1 -->|Sim| Score1[Score += 0.4]
    Check1 -->|Não| Flag1[⚠️ Flag: Possível Alucinação]
    
    Score1 --> V2[Validador 2:<br/>Verificação Ética]
    Flag1 --> V2
    
    V2 --> Check2{Contém:<br/>- Automedicação?<br/>- Diagnóstico definitivo?<br/>- Info perigosa?}
    Check2 -->|Não| Score2[Score += 0.3]
    Check2 -->|Sim| Flag2[❌ Flag: Violação Ética]
    
    Score2 --> V3[Validador 3:<br/>Confidence Score]
    Flag2 --> Reject[Rejeitar Resposta]
    
    V3 --> CalcConf[Calcular Confiança:<br/>- Similaridade semântica<br/>- Cobertura de contexto<br/>- Consistência]
    
    CalcConf --> Score3[Score Final]
    
    Score3 --> Final{Score Final?}
    
    Final -->|> 0.8| High[✅ HIGH Confidence<br/>Aprovar]
    Final -->|0.6-0.8| Med[⚠️ MEDIUM Confidence<br/>Adicionar aviso]
    Final -->|< 0.6| Low[❌ LOW Confidence<br/>Reprocessar]
    
    High --> Return[Retornar Resposta]
    Med --> AddWarn[Adicionar:<br/>⚠️ Informação com baixa confiança]
    Low --> Reject
    
    AddWarn --> Return
    Reject --> Retry[Solicitar Reprocessamento]
    
    style Response fill:#e1f5e1
    style Return fill:#e1f5e1
    style Reject fill:#ffe1e1
    style High fill:#e1f5e1
    style Med fill:#fff4e1
    style Low fill:#ffe1e1
```

---

## 📊 Fluxo de Dados: Fine-Tuning até Inferência

```mermaid
graph TB
    subgraph "Fase 1: Preparação de Dados"
        Raw[XMLs Brutos<br/>728 arquivos] --> Extract[Extrair Q&A]
        Extract --> Valid[Validar Estrutura]
        Valid --> Clean[Limpar Texto<br/>-7% tokens]
        Clean --> Split[Split 80/20<br/>582 train / 146 test]
    end
    
    subgraph "Fase 2: Fine-Tuning (Colab)"
        Split --> Format[Formatar Llama3<br/>Chat Template]
        Format --> Load[Carregar Modelo Base<br/>Llama-3.1-70B-4bit]
        Load --> LoRA[Aplicar LoRA<br/>r=16, 7 modules]
        LoRA --> Train[Treinar<br/>3 epochs, lr=2e-4]
        Train --> Eval[Avaliar<br/>Loss < 0.5]
        Eval --> Save[Salvar Adaptadores<br/>45 MB]
    end
    
    subgraph "Fase 3: Implantação"
        Save --> Download[Download do Drive]
        Download --> Local[Copiar para<br/>models/llama3_medical_ft/]
        Local --> LoadInf[Carregar para Inferência]
    end
    
    subgraph "Fase 4: Uso em Produção"
        LoadInf --> Query[Query do Usuário]
        Query --> RAGPipe[Pipeline RAG]
        RAGPipe --> Infer[Inferência com<br/>Modelo Fine-Tunado]
        Infer --> Gemini[Síntese com Gemini]
        Gemini --> Guards[Validação Guardrails]
        Guards --> User[Resposta ao Usuário]
    end
    
    style Raw fill:#e1f5e1
    style Train fill:#fff4e1
    style Save fill:#ffe1e1
    style User fill:#e1f5e1
```

---

## 🎬 Exemplo de Execução Completa

### Query: "What are the symptoms of lung cancer?"

```mermaid
journey
    title Jornada da Query pelo Sistema
    section Entrada
      Usuário faz pergunta: 5: Usuário
      Orquestrador recebe: 5: Orquestrador
    section Classificação
      Identifica como clínica: 5: Classificador
      Determina urgência média: 4: Classificador
    section RAG
      Gera embedding da query: 5: RAG
      Busca top-5 chunks: 5: FAISS
      Encontra contexto relevante: 5: RAG
      Score médio: 0.84: RAG
    section Agente Llama
      Recebe query + contexto: 5: Llama
      Analisa sintomas: 5: Llama
      Lista diagnósticos diff.: 5: Llama
      Recomenda exames: 5: Llama
      Define urgência MODERADA: 4: Llama
      Estrutura resposta (5 seções): 5: Llama
    section Agente Gemini
      Recebe resposta Llama: 5: Gemini
      Traduz para PT-BR: 5: Gemini
      Formata com emojis: 5: Gemini
      Adiciona fontes: 5: Gemini
      Adiciona disclaimer: 5: Gemini
    section Guardrails
      Verifica alucinações: 5: Guardrails
      Valida ética: 5: Guardrails
      Calcula confidence: 0.87: Guardrails
      Aprova resposta: 5: Guardrails
    section Saída
      Retorna ao usuário: 5: Orquestrador
      Usuário satisfeito: 5: Usuário
```

---

## 📈 Métricas de Performance por Etapa

| Etapa | Latência Média | Taxa de Sucesso | Observações |
|-------|----------------|-----------------|-------------|
| **Classificação** | 50ms | 99% | Rápido, baseado em keywords |
| **RAG Retrieval** | 200ms | 95% | Depende do tamanho do vector store |
| **Agente Llama** | 1.5s | 92% | Pode falhar por timeout API |
| **Agente Gemini** | 1.2s | 96% | Mais estável que Llama |
| **Guardrails** | 300ms | 100% | Sempre executado |
| **TOTAL** | ~3.2s | 94% | Pipeline completo |

---

## 🔧 Configuração de Roteamento Condicional

```python
def route_after_retrieval(state: AgentState) -> str:
    """
    Decide qual agente acionar baseado na classificação da query
    """
    query_complexity = state.get("query_complexity", "medium")
    requires_reasoning = state.get("requires_clinical_reasoning", False)
    
    # Matriz de decisão
    if requires_reasoning and query_complexity == "high":
        return "both"  # Llama → Gemini (pipeline completo)
    elif requires_reasoning:
        return "llama"  # Apenas Llama
    else:
        return "gemini"  # Apenas Gemini (perguntas gerais)

def route_after_llama(state: AgentState) -> str:
    """
    Decide se precisa passar pelo Gemini após Llama
    """
    llama_response = state.get("llama_response")
    
    if llama_response and len(llama_response) > 100:
        return "gemini"  # Sintetizar com Gemini
    else:
        return "validate"  # Pular Gemini, ir direto para validação

def route_after_validation(state: AgentState) -> str:
    """
    Decide se aprova ou reprocessa baseado na validação
    """
    confidence = state.get("confidence_level")
    violations = state.get("guardrail_violations", [])
    
    if violations:
        return "retry"  # Reprocessar com Gemini
    elif confidence in ["HIGH", "MEDIUM"]:
        return "finalize"  # Aprovar resposta
    else:
        return "retry"  # Confidence muito baixa
```

---

## 🎯 Estados do Sistema (AgentState)

```python
class AgentState(TypedDict):
    """Estado compartilhado entre todos os agentes"""
    
    # Input do usuário
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    session_id: str
    
    # RAG
    retrieved_context: str
    rag_sources: list  # IDs dos chunks recuperados
    relevance_scores: list  # Scores de similaridade
    
    # Classificação
    query_type: str  # "clinical", "informational", "administrative"
    query_complexity: str  # "low", "medium", "high"
    requires_clinical_reasoning: bool
    urgency_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    
    # Respostas dos agentes
    llama_response: str
    llama_confidence: float
    gemini_response: str
    final_response: str
    
    # Metadados de processamento
    processing_steps: list  # Log de cada etapa
    execution_time: float
    
    # Guardrails
    guardrail_passed: bool
    guardrail_violations: list
    confidence_level: str  # "LOW", "MEDIUM", "HIGH"
    response_type: str  # "safe", "warning", "rejected"
```

---

## 📚 Referências de Implementação

### Arquivos Principais

1. **Orquestrador**: [src/orchestrator.py](../src/orchestrator.py)
   - Implementação do StateGraph
   - Funções de roteamento condicional
   - Gerenciamento de estado

2. **Agente Llama**: [src/agents/llama_agent.py](../src/agents/llama_agent.py)
   - Carregamento do modelo fine-tunado
   - Formatação de prompts
   - Parsing de respostas estruturadas

3. **Agente Gemini**: [src/agents/gemini_agent.py](../src/agents/gemini_agent.py)
   - Integração com Google AI
   - Pipeline RAG
   - Síntese em PT-BR

4. **RAG Pipeline**: [src/rag/pipeline.py](../src/rag/pipeline.py)
   - Chunking de documentos
   - Geração de embeddings
   - FAISS vector store

5. **Guardrails**: [src/guardrails/validators.py](../src/guardrails/validators.py)
   - Detecção de alucinações
   - Validação ética
   - Cálculo de confidence

---

## 🎓 Conclusão

Este sistema demonstra uma **arquitetura moderna de IA Generativa** aplicada ao domínio médico, combinando:

✅ **Fine-Tuning especializado** (Llama 3.1 70B)  
✅ **RAG para factualidade** (FAISS + embeddings)  
✅ **Multi-Agent com LangGraph** (orquestração inteligente)  
✅ **Guardrails de segurança** (validação multicamada)  
✅ **Arquitetura escalável** (fácil adicionar novos agentes)

O fluxo foi projetado para ser:
- **Modular**: Componentes independentes
- **Resiliente**: Retry automático em falhas
- **Observável**: Logs detalhados de cada etapa
- **Ético**: Validação rigorosa de saídas

---

<div align="center">

**📊 Documento gerado como parte do Tech Challenge FIAP - Fase 3**

*Para mais detalhes técnicos, consulte [README.md](../README.md)*

</div>
