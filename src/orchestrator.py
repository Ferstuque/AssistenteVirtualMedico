"""
Orquestração Multi-Agente usando LangGraph
Coordena os agentes Llama (raciocínio) e Gemini (RAG/insights)
"""
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from datetime import datetime
from src.agents.llama_agent import LlamaReasoningAgent
from src.agents.gemini_agent import GeminiRAGAgent
from src.guardrails.validators import GuardrailValidator, ResponseType, ConfidenceLevel
from src.utils.logger import get_logger
from src.rag.pipeline import MedicalRAGPipeline


class AgentState(TypedDict):
    """
    Estado compartilhado entre os agentes
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    session_id: str
    
    # RAG
    retrieved_context: str
    rag_sources: list
    
    # Decisões
    requires_clinical_reasoning: bool
    urgency_level: str
    
    # Respostas
    llama_response: str
    gemini_response: str
    final_response: str
    
    # Metadados
    processing_steps: list
    confidence_level: str
    response_type: str
    
    # Guardrails
    guardrail_passed: bool
    guardrail_violations: list


class MedicalAssistantOrchestrator:
    """
    Orquestrador principal do assistente médico
    Coordena Llama (raciocínio) e Gemini (RAG/insights)
    """
    
    def __init__(
        self,
        llama_agent: LlamaReasoningAgent,
        gemini_agent: GeminiRAGAgent,
        rag_pipeline: MedicalRAGPipeline
    ):
        """
        Inicializa orquestrador
        
        Args:
            llama_agent: Agente Llama para raciocínio
            gemini_agent: Agente Gemini para RAG
            rag_pipeline: Pipeline de RAG
        """
        self.llama_agent = llama_agent
        self.gemini_agent = gemini_agent
        self.rag_pipeline = rag_pipeline
        self.validator = GuardrailValidator()
        self.logger = get_logger()
        
        # Configurar RAG no Gemini
        self.gemini_agent.set_rag_pipeline(rag_pipeline)
        
        # Construir graph
        self.graph = self._build_graph()
        
        print("✓ Orquestrador Multi-Agente inicializado")
    
    def _build_graph(self) -> StateGraph:
        """Constrói o grafo de execução com LangGraph"""
        
        workflow = StateGraph(AgentState)
        
        # Adicionar nós (etapas do workflow)
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("llama_reasoning", self._llama_reasoning)
        workflow.add_node("gemini_response", self._gemini_response)
        workflow.add_node("validate_response", self._validate_response)
        workflow.add_node("finalize", self._finalize)
        
        # Definir fluxo
        workflow.set_entry_point("classify_query")
        
        # Classificação -> Retrieval
        workflow.add_edge("classify_query", "retrieve_context")
        
        # Retrieval -> Decisão condicional
        workflow.add_conditional_edges(
            "retrieve_context",
            self._route_after_retrieval,
            {
                "llama": "llama_reasoning",
                "gemini": "gemini_response",
                "both": "llama_reasoning"  # Se ambos, começa com Llama
            }
        )
        
        # Llama -> Gemini (quando ambos necessários)
        workflow.add_conditional_edges(
            "llama_reasoning",
            self._route_after_llama,
            {
                "gemini": "gemini_response",
                "validate": "validate_response"
            }
        )
        
        # Gemini -> Validação
        workflow.add_edge("gemini_response", "validate_response")
        
        # Validação -> Decisão
        workflow.add_conditional_edges(
            "validate_response",
            self._route_after_validation,
            {
                "finalize": "finalize",
                "retry": "gemini_response"  # Se falhar validação, reprocessar
            }
        )
        
        # Finalizar -> END
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Classifica tipo de query do usuário"""
        query = state["user_query"].lower()
        
        # Palavras-chave que indicam necessidade de raciocínio clínico
        clinical_keywords = [
            "sintoma", "dor", "febre", "diagnóstico", "tratamento",
            "exame", "urgente", "grave", "emergência",
            # English keywords
            "symptom", "pain", "fever", "diagnosis", "treatment",
            "exam", "urgent", "severe", "emergency", "cancer", "disease"
        ]
        
        requires_reasoning = any(keyword in query for keyword in clinical_keywords)
        
        state["requires_clinical_reasoning"] = requires_reasoning
        state["processing_steps"].append({
            "step": "classify_query",
            "result": "clinical_reasoning" if requires_reasoning else "general_info",
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.log_request(
            user_query=state["user_query"],
            session_id=state.get("session_id"),
            metadata={"requires_reasoning": requires_reasoning}
        )
        
        return state
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """Recupera contexto do RAG"""
        retrieval_info = self.rag_pipeline.retrieve_with_metadata(
            query=state["user_query"],
            k=5
        )
        
        state["retrieved_context"] = self.rag_pipeline.create_context_from_retrieval(retrieval_info)
        state["rag_sources"] = [chunk['chunk_id'] for chunk in retrieval_info['chunks']]
        
        state["processing_steps"].append({
            "step": "retrieve_context",
            "num_chunks": retrieval_info['num_results'],
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.log_rag_retrieval(
            query=state["user_query"],
            retrieved_chunks=retrieval_info['chunks'],
            num_chunks=retrieval_info['num_results'],
            sources=state["rag_sources"]
        )
        
        return state
    
    def _llama_reasoning(self, state: AgentState) -> AgentState:
        """Executa raciocínio clínico com Llama"""
        result = self.llama_agent.reason_with_context(
            query=state["user_query"],
            retrieved_context=state["retrieved_context"],
            task_type="analysis"
        )
        
        state["llama_response"] = result["analysis"]
        state["urgency_level"] = "ROUTINE"  # Default
        
        state["processing_steps"].append({
            "step": "llama_reasoning",
            "processing_time": result["processing_time"],
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.log_agent_decision(
            agent_name="llama",
            decision=state["llama_response"][:200],
            metadata={"urgency": state["urgency_level"]}
        )
        
        return state
    
    def _gemini_response(self, state: AgentState) -> AgentState:
        """Gera resposta final com Gemini"""
        result = self.gemini_agent.answer_with_rag(
            user_question=state["user_query"],
            k_retrieve=5
        )
        
        state["gemini_response"] = result["response"]
        
        # Se não tem resposta do Llama, usa a do Gemini como base
        if not state.get("llama_response"):
            state["final_response"] = result["response"]
        else:
            # Combina insights de ambos os agentes
            state["final_response"] = f"{result['response']}\n\n[Análise Clínica]: {state['llama_response'][:300]}..."
        
        state["processing_steps"].append({
            "step": "gemini_response",
            "processing_time": result["processing_time"],
            "num_sources": result["num_sources"],
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    def _validate_response(self, state: AgentState) -> AgentState:
        """Valida resposta contra guardrails"""
        try:
            # Determinar tipo de resposta
            response_type = ResponseType.RECOMMENDATION if state.get("requires_clinical_reasoning") else ResponseType.INFORMATIONAL
            
            # Validar
            validated = self.validator.validate_response({
                "response_text": state["final_response"],
                "response_type": response_type,
                "confidence_level": ConfidenceLevel.MEDIUM,
                "sources": state["rag_sources"]
            })
            
            state["guardrail_passed"] = True
            state["guardrail_violations"] = []
            state["response_type"] = response_type.value
            state["confidence_level"] = ConfidenceLevel.MEDIUM.value
            
        except Exception as e:
            state["guardrail_passed"] = False
            state["guardrail_violations"] = [str(e)]
            
            self.logger.log_guardrail_violation(
                violation_type="response_validation",
                input_text=state["final_response"],
                reason=str(e)
            )
        
        state["processing_steps"].append({
            "step": "validate_response",
            "passed": state["guardrail_passed"],
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    def _finalize(self, state: AgentState) -> AgentState:
        """Finaliza e loga resposta"""
        total_time = sum(
            step.get("processing_time", 0) 
            for step in state["processing_steps"]
        )
        
        self.logger.log_response(
            response=state["final_response"],
            response_time=total_time,
            agent_used="multi_agent",
            sources=state["rag_sources"],
            metadata={
                "steps": len(state["processing_steps"]),
                "llama_used": bool(state.get("llama_response")),
                "gemini_used": bool(state.get("gemini_response"))
            }
        )
        
        return state
    
    def _route_after_retrieval(self, state: AgentState) -> str:
        """Decide roteamento após retrieval"""
        if state["requires_clinical_reasoning"]:
            return "both"  # Usa ambos os agentes
        else:
            return "gemini"  # Apenas Gemini para respostas informativas
    
    def _route_after_llama(self, state: AgentState) -> str:
        """Decide roteamento após Llama"""
        return "gemini"  # Sempre passa pelo Gemini para resposta final
    
    def _route_after_validation(self, state: AgentState) -> str:
        """Decide roteamento após validação"""
        if state["guardrail_passed"]:
            return "finalize"
        else:
            # Em caso de falha, poderia reprocessar ou retornar mensagem de erro
            # Por simplicidade, finaliza com erro
            state["final_response"] = "Desculpe, não consigo fornecer uma resposta adequada para esta questão. Por favor, consulte um profissional de saúde."
            return "finalize"
    
    def process_query(self, user_query: str, session_id: str = None) -> dict:
        """
        Processa query do usuário através do workflow completo
        
        Args:
            user_query: Pergunta do usuário
            session_id: ID da sessão (opcional)
        
        Returns:
            Dicionário com resposta e metadados
        """
        # Estado inicial
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "session_id": session_id or f"session_{datetime.now().timestamp()}",
            "retrieved_context": "",
            "rag_sources": [],
            "requires_clinical_reasoning": False,
            "urgency_level": "ROUTINE",
            "llama_response": "",
            "gemini_response": "",
            "final_response": "",
            "processing_steps": [],
            "confidence_level": "medium",
            "response_type": "informational",
            "guardrail_passed": True,
            "guardrail_violations": []
        }
        
        # Executar workflow
        final_state = self.graph.invoke(initial_state)
        
        # Retornar resultado
        return {
            "response": final_state["final_response"],
            "llama_response": final_state.get("llama_response", ""),
            "gemini_response": final_state.get("gemini_response", ""),
            "sources": final_state["rag_sources"],
            "confidence_level": final_state["confidence_level"],
            "response_type": final_state["response_type"],
            "urgency_level": final_state["urgency_level"],
            "processing_steps": final_state["processing_steps"],
            "guardrail_passed": final_state["guardrail_passed"],
            "session_id": final_state["session_id"]
        }
    
    def analyze_patient_with_orchestrator(
        self,
        patient_id: str = None,
        patient_name: str = None,
        specific_question: str = None,
        session_id: str = None
    ) -> dict:
        """
        Analisa prontuário de paciente usando o fluxo completo do orquestrador
        (Llama para raciocínio clínico + Gemini com RAG para resposta final)
        
        Args:
            patient_id: ID do paciente para buscar prontuário
            patient_name: Nome do paciente (alternativa ao ID)
            specific_question: Pergunta específica sobre o paciente
            session_id: ID da sessão (opcional)
        
        Returns:
            Dicionário com análise completa e metadados
        """
        import time
        start_time = time.time()
        
        # Buscar dados do paciente
        patient = self.gemini_agent.search_patient(
            patient_id=patient_id,
            patient_name=patient_name
        )
        
        if not patient:
            return {
                "error": "Paciente não encontrado",
                "patient_id": patient_id,
                "patient_name": patient_name,
                "response": "Não foi possível localizar o paciente solicitado no sistema."
            }
        
        # Construir query contextualizada com o prontuário
        patient_context = f"""
Prontuário do Paciente:
- ID: {patient['id_paciente']}
- Nome: {patient['nome']}
- Idade: {patient['idade']}
- Histórico Clínico: {patient['historico']}

{specific_question if specific_question else 'Por favor, analise este caso clínico e forneça um prognóstico detalhado com recomendações baseadas em evidências.'}
"""
        
        # Processar através do orquestrador (fluxo completo LangChain)
        result = self.process_query(
            user_query=patient_context,
            session_id=session_id or f"patient_analysis_{patient['id_paciente']}_{datetime.now().timestamp()}"
        )
        
        # Enriquecer resultado com informações do paciente
        result.update({
            "patient_id": patient['id_paciente'],
            "patient_name": patient['nome'],
            "patient_age": patient['idade'],
            "patient_history": patient['historico'],
            "processing_time": time.time() - start_time,
            "model": "Llama3 (reasoning) + Gemini-Pro (RAG synthesis)",
            "rag_context_used": len(result.get('sources', [])) > 0,
            "rag_sources": [
                {
                    "chunk_id": source,
                    "source": "MedQuAD Medical Database",
                    "score": "N/A"  # Score seria obtido do RAG pipeline
                }
                for source in result.get('sources', [])
            ]
        })
        
        # Log da análise
        self.logger.log_request(
            user_query=f"Patient Analysis: {patient['id_paciente']}",
            session_id=result['session_id'],
            metadata={
                "patient_id": patient['id_paciente'],
                "analysis_type": "full_orchestrator_flow",
                "llama_used": bool(result.get('llama_response')),
                "rag_sources": len(result.get('sources', []))
            }
        )
        
        return result


def main():
    """Teste do orquestrador"""
    print("⚠️ Para testar o orquestrador, é necessário:")
    print("  1. Configurar HUGGINGFACEHUB_API_TOKEN no .env")
    print("  2. Configurar GOOGLE_API_KEY no .env")
    print("  3. Construir o vector store com dados")
    print("  4. Executar o sistema completo")


if __name__ == "__main__":
    main()
