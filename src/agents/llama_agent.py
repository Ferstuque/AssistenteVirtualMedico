"""
Agente 1: Raciocínio e Decisão usando Llama (HuggingFace)
Fine-tuning e inferência clínica
"""
from typing import Dict, Optional, List
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from langchain_core.messages import HumanMessage, SystemMessage
import os
from datetime import datetime


class LlamaReasoningAgent:
    """
    Agente de raciocínio clínico usando Llama
    Responsável por análise, inferência e decisões clínicas
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_token: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 256
    ):
        """
        Inicializa agente Llama
        
        Args:
            model_name: Nome do modelo no HuggingFace (usa LLAMA_MODEL_NAME do .env se não fornecido)
            api_token: Token de API do HuggingFace
            temperature: Temperatura para geração
            max_new_tokens: Número máximo de tokens a gerar
        """
        self.model_name = model_name or os.getenv("LLAMA_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.api_token = api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        # Provider será escolhido automaticamente pelo HuggingFace
        self.provider = None
        # Usar task conversational (exigido pelos providers automáticos)
        self.use_chat_wrapper = True
        
        print(f"   ℹ️ Usando HuggingFace Inference API com task conversational (ChatHuggingFace wrapper)")
        
        if not self.api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN não configurado")
        
        print(f"🤖 Inicializando Llama Reasoning Agent...")
        print(f"   Modelo: {self.model_name}")
        if self.provider:
            print(f"   Provider: {self.provider}")
        print(f"   Chat wrapper: {'ativado' if self.use_chat_wrapper else 'desativado'}")
        
        # Configurar LLM base (SEM especificar task - deixa o HF decidir)
        # ⚠️ IMPORTANTE: Com do_sample=False, NÃO use temperature, top_k ou top_p
        self.llm = HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_token,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Determinístico - greedy decoding
            repetition_penalty=1.03,  # Evita repetições
        )
        print(f"   ✓ Modelo {self.model_name} inicializado (direct endpoint)")
        
        # Template de prompt para raciocínio clínico (simplificado)
        self.reasoning_template = PromptTemplate(
            input_variables=["patient_info", "symptoms", "context"],
            template="""Medical assistant - clinical analysis.

Patient: {patient_info}
Symptoms: {symptoms}
Context: {context}

Provide:
1. Clinical analysis
2. Differential diagnoses
3. Recommended investigations
4. Medical guidance

Note: No direct prescriptions. Refer to physician.

Analysis:"""
        )
        
        # Template para decisão sobre encaminhamento (simplificado)
        self.decision_template = PromptTemplate(
            input_variables=["case_summary", "severity_indicators"],
            template="""Medical triage specialist.

Case: {case_summary}
Severity: {severity_indicators}

Classify urgency:
- URGENT: Immediate care
- PRIORITY: 24-48h
- ROUTINE: Regular appointment
- GUIDANCE: General advice

Decision:"""
        )
        
        print("✓ Llama Agent inicializado")

    def _invoke_llm(self, prompt: str) -> str:
        """Executa o LLM diretamente (sem wrapper de chat) e retorna a resposta."""
        try:
            # Invocar diretamente o endpoint base
            response = self.llm.invoke(prompt)
            
            # Normalizar resposta para string
            if hasattr(response, "content"):
                return response.content
            if isinstance(response, dict) and "text" in response:
                return response["text"]
            return response if isinstance(response, str) else str(response)
        except BadRequestError as exc:
            print(f"⚠️ Llama Agent - erro HTTP 400: {exc}")
            raise
        except Exception as exc:
            print(f"⚠️ Llama Agent - erro durante chamada ao modelo: {exc}")
            raise
    
    def analyze_case(
        self,
        patient_info: str,
        symptoms: str,
        context: str = ""
    ) -> Dict:
        """
        Analisa caso clínico e fornece raciocínio
        
        Args:
            patient_info: Informações do paciente (anonimizadas)
            symptoms: Sintomas apresentados
            context: Contexto adicional
        
        Returns:
            Dicionário com análise
        """
        start_time = datetime.now()
        
        # Formatar prompt e executar
        prompt = self.reasoning_template.format(
            patient_info=patient_info,
            symptoms=symptoms,
            context=context
        )
        
        result_text = self._invoke_llm(prompt)
        
        end_time = datetime.now()
        
        return {
            "agent": "llama_reasoning",
            "analysis": result_text,
            "timestamp": datetime.now().isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "model": self.model_name
        }
    
    def make_decision(
        self,
        case_summary: str,
        severity_indicators: List[str]
    ) -> Dict:
        """
        Toma decisão sobre encaminhamento/urgência
        
        Args:
            case_summary: Resumo do caso
            severity_indicators: Lista de indicadores de gravidade
        
        Returns:
            Dicionário com decisão
        """
        start_time = datetime.now()
        
        severity_text = "\n".join(f"- {indicator}" for indicator in severity_indicators)
        
        prompt = self.decision_template.format(
            case_summary=case_summary,
            severity_indicators=severity_text
        )
        
        result_text = self._invoke_llm(prompt)
        
        end_time = datetime.now()
        
        decision_text = result_text

        # Extrair nível de urgência
        urgency = "ROTINA"
        if "URGENTE" in decision_text.upper():
            urgency = "URGENTE"
        elif "PRIORITÁRIO" in decision_text.upper() or "PRIORITARIO" in decision_text.upper():
            urgency = "PRIORITÁRIO"
        elif "ORIENTAÇÃO" in decision_text.upper() or "ORIENTACAO" in decision_text.upper():
            urgency = "ORIENTAÇÃO"
        
        return {
            "agent": "llama_reasoning",
            "decision": decision_text,
            "urgency_level": urgency,
            "timestamp": datetime.now().isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "model": self.model_name
        }
    
    def reason_with_context(
        self,
        query: str,
        retrieved_context: str,
        task_type: str = "analysis"
    ) -> Dict:
        """
        Raciocínio com contexto do RAG
        
        Args:
            query: Pergunta/caso
            retrieved_context: Contexto recuperado do RAG
            task_type: Tipo de tarefa (analysis, decision)
        
        Returns:
            Resultado do raciocínio
        """
        if task_type == "decision":
            return self.make_decision(
                case_summary=query,
                severity_indicators=[retrieved_context]
            )
        else:
            return self.analyze_case(
                patient_info="Informações anonimizadas",
                symptoms=query,
                context=retrieved_context
            )


def main():
    """Teste do agente Llama"""
    
    # Nota: Requer HUGGINGFACEHUB_API_TOKEN configurado
    try:
        agent = LlamaReasoningAgent()
        
        # Teste de análise
        result = agent.analyze_case(
            patient_info="Paciente adulto, sem comorbidades conhecidas",
            symptoms="Febre persistente há 3 dias, tosse seca, fadiga",
            context="Período de circulação de vírus respiratórios"
        )
        
        print("\n📊 RESULTADO DA ANÁLISE:")
        print(f"Agente: {result['agent']}")
        print(f"Tempo: {result['processing_time']:.2f}s")
        print(f"Análise: {result['analysis']}")
        
    except Exception as e:
        print(f"⚠️ Erro ao testar agente: {e}")
        print("Certifique-se de que HUGGINGFACEHUB_API_TOKEN está configurado no .env")


if __name__ == "__main__":
    main()
