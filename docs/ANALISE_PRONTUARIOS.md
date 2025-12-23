# Análise de Prontuários de Pacientes

## Visão Geral

O Assistente Virtual Médico agora inclui funcionalidade para análise de prontuários de pacientes do hospital, permitindo que médicos obtenham prognósticos baseados no histórico clínico dos pacientes.

## Características

✅ **25 Prontuários Fictícios** - Dataset com casos clínicos diversos para demonstração  
✅ **Busca por ID ou Nome** - Localização rápida de pacientes  
✅ **Análise de Prognóstico** - IA analisa histórico e sugere diagnósticos  
✅ **Perguntas Específicas** - Faça perguntas direcionadas sobre cada paciente  
✅ **Recomendações de Exames** - Sugestões de exames complementares  
✅ **Avaliação de Riscos** - Identificação de fatores de risco

## Estrutura do Dataset

O arquivo `data/processed/prontuarios_pacientes.json` contém:

```json
{
  "metadata": {
    "description": "Prontuários fictícios de pacientes do hospital para análise de prognóstico",
    "criado_em": "2025-12-20",
    "total_registros": 25,
    "aviso": "DADOS FICTÍCIOS - Apenas para fins de demonstração e teste"
  },
  "pacientes": [
    {
      "id_paciente": "brcp230442",
      "nome": "José",
      "idade": "47 anos",
      "historico": "O paciente deu entrada no hospital em 22 de maio de 2019..."
    }
  ]
}
```

## Casos Clínicos Incluídos

O dataset inclui 25 casos diversos:

1. **Câncer de Pulmão** - Paciente fumante com histórico familiar
2. **Câncer de Fígado** - Sintomas abdominais e icterícia
3. **Diabetes Tipo 2** - Obesidade e histórico familiar
4. **Hipertensão** - Pressão alta e histórico de infarto na família
5. **Asma/DPOC** - Dificuldades respiratórias crônicas
6. **Insuficiência Renal** - Infecções urinárias recorrentes
7. **Infarto do Miocárdio** - Dor torácica e fatores de risco
8. **Anemia Falciforme** - Crises vaso-oclusivas
9. **Parkinson** - Sintomas motores progressivos
10. **Câncer de Mama** - Nódulo palpável e histórico familiar
11. **Cirrose Hepática** - Etilismo crônico
12. **Trombocitopenia** - Plaquetas baixas
13. **Pneumonia** - Infecção pulmonar
14. **Epilepsia/Tumor Cerebral** - Convulsões e alterações visuais
15. **Artrite Reumatoide** - Dor articular crônica
16. **Hipertireoidismo** - Sintomas metabólicos
17. **Doença Inflamatória Intestinal** - Diarreia crônica
18. **AVC** - Fraqueza súbita e dificuldade de fala
19. **Trauma Abdominal** - Acidente com laceração esplênica
20. **Pré-eclâmpsia** - Gestação com hipertensão
21. **Litíase Renal** - Cálculo renal recorrente
22. **Transtornos de Ansiedade** - Pânico e depressão
23. **Mieloma Múltiplo** - Anemia e dor óssea
24. **Mononucleose** - Infecção viral
25. **Insuficiência Cardíaca** - Dispneia e edema

## Como Usar

### 1. Importar o Agente

```python
from src.agents.gemini_agent import GeminiRAGAgent

# Inicializar agente
agent = GeminiRAGAgent()
```

### 2. Listar Pacientes Disponíveis

```python
# Listar todos os pacientes
patients = agent.list_all_patients()

for patient in patients:
    print(f"ID: {patient['id_paciente']}")
    print(f"Nome: {patient['nome']}")
    print(f"Idade: {patient['idade']}\n")
```

### 3. Buscar um Paciente Específico

```python
# Buscar por ID
patient = agent.search_patient(patient_id="brcp230442")

# Ou buscar por nome
patient = agent.search_patient(patient_name="José")

if patient:
    print(f"Encontrado: {patient['nome']}")
    print(f"Histórico: {patient['historico']}")
```

### 4. Análise Completa do Prontuário

```python
# Análise geral do paciente
result = agent.analyze_patient_prognosis(patient_id="brcp230442")

print(f"Paciente: {result['patient_name']}")
print(f"Análise:\n{result['response']}")
print(f"Tempo: {result['processing_time']:.2f}s")
```

### 5. Perguntas Específicas

```python
# Fazer uma pergunta específica sobre o paciente
result = agent.analyze_patient_prognosis(
    patient_name="Maria",
    specific_question="Quais exames você recomendaria para confirmar o diagnóstico?"
)

print(result['response'])
```

### 6. Integração com o Sistema Principal

```python
from src.main import MedicalAssistant

# Inicializar assistente
assistant = MedicalAssistant()

# O Gemini Agent já terá os prontuários carregados
gemini_agent = assistant.gemini_agent

# Usar as funcionalidades
result = gemini_agent.analyze_patient_prognosis(
    patient_id="brdb450123",
    specific_question="Avalie o risco cardiovascular deste paciente"
)
```

## Exemplos de Perguntas

Você pode fazer perguntas como:

- "Qual o diagnóstico mais provável baseado nos sintomas?"
- "Quais exames complementares você recomenda?"
- "Quais são os principais fatores de risco deste paciente?"
- "Qual a probabilidade de [doença específica]?"
- "Que medidas preventivas você sugere?"
- "Como avaliar a progressão da doença?"
- "Há indicação de internação ou tratamento ambulatorial?"

## Executar Demonstração

Para ver exemplos práticos, execute:

```bash
python scripts/test_prontuarios.py
```

Este script demonstra:
- Listagem de pacientes
- Análise completa de prontuário
- Perguntas específicas
- Tratamento de erros

## Formato de Resposta

O método `analyze_patient_prognosis()` retorna um dicionário:

```python
{
    "agent": "gemini_rag",
    "response": "Análise detalhada do prontuário...",
    "patient_id": "brcp230442",
    "patient_name": "José",
    "patient_age": "47 anos",
    "analysis_type": "prognosis",
    "timestamp": "2025-12-20T...",
    "processing_time": 2.34,
    "model": "gemini-1.5-flash",
    "language": "pt-BR"
}
```

## Diretrizes da IA

A IA foi configurada para:

✅ Analisar o histórico de forma detalhada  
✅ Identificar fatores de risco  
✅ Sugerir possíveis diagnósticos  
✅ Recomendar exames complementares  
✅ Usar linguagem técnica mas compreensível  
✅ Manter confidencialidade dos dados  

❌ **NUNCA** prescrever medicamentos  
❌ **SEMPRE** reforçar necessidade de avaliação médica presencial  

## Segurança e Privacidade

⚠️ **IMPORTANTE:**
- Todos os dados são **FICTÍCIOS**
- Criados apenas para demonstração
- Não usar com dados reais de pacientes
- Sempre anonimizar dados reais antes de processar

## Adicionar Novos Prontuários

Para adicionar novos pacientes ao dataset:

1. Edite `data/processed/prontuarios_pacientes.json`
2. Adicione um novo objeto no array `pacientes`:

```json
{
  "id_paciente": "brxx999999",
  "nome": "Nome do Paciente",
  "idade": "XX anos",
  "historico": "Descrição detalhada do histórico clínico..."
}
```

3. Atualize o campo `total_registros` nos metadados
4. Reinicie o agente para carregar os novos dados

## Limitações

- Dataset limitado a 25 pacientes fictícios
- Não substitui avaliação médica real
- IA pode ter limitações em casos muito complexos
- Sempre validar recomendações com profissional qualificado

## Próximos Passos

Possíveis melhorias futuras:

- [ ] Integração com banco de dados real (com devidas autorizações)
- [ ] Sistema de histórico de consultas por paciente
- [ ] Comparação de evolução clínica ao longo do tempo
- [ ] Alertas automáticos para casos críticos
- [ ] Geração de relatórios médicos estruturados
- [ ] Suporte a imagens médicas (Raio-X, TC, RM)

## Suporte

Para dúvidas ou problemas:
1. Verifique se o arquivo `prontuarios_pacientes.json` existe
2. Confirme que a GOOGLE_API_KEY está configurada
3. Execute o script de teste: `python scripts/test_prontuarios.py`
4. Verifique os logs de erro do agente

## Referências

- Dataset: `data/processed/prontuarios_pacientes.json`
- Agente: `src/agents/gemini_agent.py`
- Script de teste: `scripts/test_prontuarios.py`
- Documentação: Este arquivo
