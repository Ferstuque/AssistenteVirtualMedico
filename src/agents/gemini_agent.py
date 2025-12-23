"""
Agente 2: RAG e Insights usando Google Gemini
Geração de respostas com retrieval-augmented generation
"""
from typing import Dict, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
from pathlib import Path
from datetime import datetime
from src.rag.pipeline import MedicalRAGPipeline


class GeminiRAGAgent:
    """
    Agente de insights e respostas usando Gemini com RAG
    Responsável por buscar conhecimento e gerar respostas assertivas
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        rag_pipeline: Optional[MedicalRAGPipeline] = None
    ):
        """
        Inicializa agente Gemini
        
        Args:
            model_name: Nome do modelo Gemini
            api_key: API key do Google
            temperature: Temperatura para geração
            rag_pipeline: Pipeline RAG (opcional)
        """
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY não configurado")
        
        print(f"🤖 Inicializando Gemini RAG Agent...")
        print(f"   Modelo: {self.model_name}")
        
        # Configurar LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
        # RAG Pipeline
        self.rag_pipeline = rag_pipeline
        
        # Carregar prontuários de pacientes
        self.prontuarios = self._load_patient_records()
        
        # Template de prompt para resposta com RAG
        self.rag_response_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente médico virtual especializado em fornecer informações precisas e baseadas em evidências para médicos.

DIRETRIZES IMPORTANTES:
1. Sempre responda em PORTUGUÊS (PT-BR)
2. Baseie suas respostas EXCLUSIVAMENTE no contexto fornecido
3. NUNCA prescreva medicamentos diretamente ao paciente
4. Use linguagem clara e acessível, mas precisa
5. SEMPRE cite as fontes utilizadas no formato [Fonte X] onde X é o número da fonte
6. Para cada afirmação importante, indique a fonte entre colchetes
7. Se não souber a resposta com base no contexto, diga claramente
8. Mantenha todas as informações pessoais anonimizadas
9. Mantenha temperatura baixa (0.1) para respostas consistentes e confiáveis

FORMATO DA RESPOSTA:
- Inicie com 'Prezado Especialista oncológico,' ou similar,
- Resposta clara e objetiva
- Citação explícita das fontes [Fonte 1], [Fonte 2], etc.
- Ao final, liste as fontes utilizadas
- Se aplicável, recomendação de consulta médica

EXEMPLO:
"A diabetes tipo 2 é caracterizada por resistência à insulina [Fonte 1]. Os principais sintomas incluem sede excessiva e fadiga [Fonte 2].

Fontes consultadas:
- Fonte 1: [descrição da fonte]
- Fonte 2: [descrição da fonte]"""),
            ("human", """CONTEXTO RECUPERADO:
{retrieved_context}

PERGUNTA DO USUÁRIO:
{user_question}

Por favor, forneça uma resposta completa e informativa em português, citando SEMPRE as fontes utilizadas.""")
        ])
        
        # Template para perguntas sem contexto
        self.general_response_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente médico virtual.

Responda em PORTUGUÊS (PT-BR) de forma educada e informativa.
Se a pergunta for sobre saúde, sempre recomende consultar um profissional.
NUNCA prescreva medicamentos."""),
            ("human", "{question}")
        ])
        
        # Template para análise de prontuário
        self.prognosis_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente médico especializado em análise de prontuários e prognósticos.

DIRETRIZES IMPORTANTES:
1. Analise o histórico do paciente de forma detalhada e profissional
2. Identifique fatores de risco e sinais de alerta CITANDO as partes específicas do prontuário
3. Sugira possíveis diagnósticos baseados nos sintomas e histórico
4. Recomende exames complementares quando apropriado, com justificativa
5. SEMPRE reforce que o diagnóstico final deve ser feito por médico presencial
6. Use linguagem técnica mas compreensível
7. Mantenha a confidencialidade e anonimização dos dados
8. NUNCA prescreva medicamentos ou tratamentos específicos
9. Mantenha temperatura baixa (0.1) para análises consistentes e confiáveis
10. Referencie EXPLICITAMENTE trechos do prontuário ao fazer afirmações

FORMATO DA RESPOSTA:
📋 **Resumo do Caso**
[Breve resumo destacando pontos principais do histórico]

🔍 **Análise do Histórico Clínico**
[Análise detalhada citando trechos relevantes do prontuário]

⚠️ **Fatores de Risco Identificados**
- [Fator 1] - [referência ao prontuário]
- [Fator 2] - [referência ao prontuário]

🩺 **Hipóteses Diagnósticas**
[Possíveis diagnósticos com base na análise]

🧪 **Exames Complementares Sugeridos**
- [Exame 1]: [justificativa]
- [Exame 2]: [justificativa]

💡 **Recomendações**
[Orientações gerais baseadas no caso]

⚕️ **Nota Importante**
Esta é uma análise assistida por IA. O diagnóstico definitivo e conduta devem ser determinados por médico em avaliação presencial."""),
            ("human", """PRONTUÁRIO DO PACIENTE:
{patient_record}

CONTEXTO MÉDICO ADICIONAL (se disponível):
{medical_context}

PERGUNTA/SOLICITAÇÃO:
{question}

Por favor, forneça uma análise detalhada considerando o histórico do paciente e citando as partes relevantes do prontuário.""")
        ])
        
        print("✓ Gemini Agent inicializado")
        print(f"   Prontuários carregados: {len(self.prontuarios)}")
    
    def _load_patient_records(self) -> List[Dict]:
        """
        Carrega prontuários de pacientes do arquivo JSON
        
        Returns:
            Lista de dicionários com dados dos pacientes
        """
        try:
            # Caminho para o arquivo de prontuários
            current_dir = Path(__file__).parent.parent.parent
            prontuarios_path = current_dir / "data" / "processed" / "prontuarios_pacientes.json"
            
            if not prontuarios_path.exists():
                print(f"⚠️  Arquivo de prontuários não encontrado: {prontuarios_path}")
                return []
            
            with open(prontuarios_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('pacientes', [])
                
        except Exception as e:
            print(f"⚠️  Erro ao carregar prontuários: {e}")
            return []
    
    def search_patient(self, patient_id: str = None, patient_name: str = None) -> Optional[Dict]:
        """
        Busca um paciente pelo ID ou nome
        
        Args:
            patient_id: ID do paciente
            patient_name: Nome do paciente
        
        Returns:
            Dicionário com dados do paciente ou None se não encontrado
        """
        if not self.prontuarios:
            return None
        
        for patient in self.prontuarios:
            if patient_id and patient.get('id_paciente', '').lower() == patient_id.lower():
                return patient
            if patient_name and patient.get('nome', '').lower() == patient_name.lower():
                return patient
        
        return None
    
    def list_all_patients(self) -> List[Dict]:
        """
        Lista todos os pacientes disponíveis
        
        Returns:
            Lista de dicionários com resumo dos pacientes
        """
        return [{
            'id_paciente': p.get('id_paciente'),
            'nome': p.get('nome'),
            'idade': p.get('idade')
        } for p in self.prontuarios]
    
    def analyze_patient_prognosis(
        self,
        patient_id: str = None,
        patient_name: str = None,
        specific_question: str = None,
        include_rag_context: bool = True
    ) -> Dict:
        """
        Analisa o prontuário de um paciente e fornece prognóstico
        
        Args:
            patient_id: ID do paciente
            patient_name: Nome do paciente
            specific_question: Pergunta específica sobre o paciente
            include_rag_context: Incluir contexto médico do RAG
        
        Returns:
            Dicionário com análise e prognóstico
        """
        start_time = datetime.now()
        
        # Buscar paciente
        patient = self.search_patient(patient_id=patient_id, patient_name=patient_name)
        
        if not patient:
            return {
                "agent": "gemini_rag",
                "error": "Paciente não encontrado",
                "patient_id": patient_id,
                "patient_name": patient_name,
                "available_patients": len(self.prontuarios),
                "timestamp": datetime.now().isoformat()
            }
        
        # Formatar prontuário
        patient_record = f"""
ID: {patient.get('id_paciente')}
Nome: {patient.get('nome')}
Idade: {patient.get('idade')}
Histórico Clínico: {patient.get('historico')}
"""
        
        # Pergunta padrão se não especificada
        if not specific_question:
            specific_question = "Forneça uma análise completa do prontuário deste paciente, incluindo possíveis diagnósticos, fatores de risco e recomendações de exames."
        
        # Buscar contexto médico do RAG se disponível
        medical_context = ""
        rag_sources = []
        
        if include_rag_context and self.rag_pipeline:
            try:
                # Criar query baseada no histórico do paciente
                rag_query = f"{patient.get('historico')} {specific_question}"
                retrieval_info = self.rag_pipeline.retrieve_with_metadata(
                    query=rag_query,
                    k=3,
                    score_threshold=0.3
                )
                
                if retrieval_info['num_results'] > 0:
                    medical_context = self.rag_pipeline.create_context_from_retrieval(retrieval_info)
                    rag_sources = [{
                        'chunk_id': chunk['chunk_id'],
                        'source': chunk['source'],
                        'score': chunk['score']
                    } for chunk in retrieval_info['chunks']]
            except Exception as e:
                print(f"⚠️ Aviso: Não foi possível buscar contexto RAG: {e}")
        
        if not medical_context:
            medical_context = "Nenhum contexto médico adicional disponível."
        
        # Gerar análise
        messages = self.prognosis_template.format_messages(
            patient_record=patient_record,
            medical_context=medical_context,
            question=specific_question
        )
        
        result = self.llm.invoke(messages)
        
        end_time = datetime.now()
        
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "agent": "gemini_rag",
            "response": response_text,
            "patient_id": patient.get('id_paciente'),
            "patient_name": patient.get('nome'),
            "patient_age": patient.get('idade'),
            "analysis_type": "prognosis",
            "timestamp": datetime.now().isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "model": self.model_name,
            "language": "pt-BR",
            "rag_sources": rag_sources,
            "rag_context_used": len(rag_sources) > 0
        }
    
    def answer_with_rag(
        self,
        user_question: str,
        k_retrieve: int = 5,
        score_threshold: float = 0.3
    ) -> Dict:
        """
        Responde pergunta usando RAG
        
        Args:
            user_question: Pergunta do usuário
            k_retrieve: Número de chunks a recuperar
            score_threshold: Threshold de similaridade
        
        Returns:
            Dicionário com resposta e metadados
        """
        start_time = datetime.now()
        
        if self.rag_pipeline is None:
            raise ValueError("RAG pipeline não configurado")
        
        # Recuperar contexto relevante
        retrieval_info = self.rag_pipeline.retrieve_with_metadata(
            query=user_question,
            k=k_retrieve,
            score_threshold=score_threshold
        )
        
        # Criar contexto formatado
        context = self.rag_pipeline.create_context_from_retrieval(retrieval_info)
        
        # Gerar resposta com chain invoke
        messages = self.rag_response_template.format_messages(
            retrieved_context=context,
            user_question=user_question
        )
        
        result = self.llm.invoke(messages)
        
        end_time = datetime.now()
        
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "agent": "gemini_rag",
            "response": response_text,
            "sources": [chunk['chunk_id'] for chunk in retrieval_info['chunks']],
            "num_sources": retrieval_info['num_results'],
            "retrieval_scores": [chunk['score'] for chunk in retrieval_info['chunks']],
            "retrieved_chunks": retrieval_info['chunks'],
            "timestamp": datetime.now().isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "model": self.model_name,
            "language": "pt-BR"
        }
    
    def answer_general(self, question: str) -> Dict:
        """
        Responde pergunta geral sem RAG
        
        Args:
            question: Pergunta do usuário
        
        Returns:
            Dicionário com resposta
        """
        start_time = datetime.now()
        
        messages = self.general_response_template.format_messages(question=question)
        result = self.llm.invoke(messages)
        
        end_time = datetime.now()
        
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "agent": "gemini_rag",
            "response": response_text,
            "sources": [],
            "num_sources": 0,
            "timestamp": datetime.now().isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "model": self.model_name,
            "language": "pt-BR"
        }
    
    def answer(
        self,
        question: str,
        context: Optional[str] = None,
        sources: Optional[List[str]] = None
    ) -> Dict:
        """
        Método de conveniência para responder com contexto já fornecido
        
        Args:
            question: Pergunta do usuário
            context: Contexto já recuperado (opcional)
            sources: Lista de IDs de chunks (opcional)
        
        Returns:
            Dicionário com resposta e metadados
        """
        start_time = datetime.now()
        
        if context:
            # Usar contexto fornecido
            messages = self.rag_response_template.format_messages(
                retrieved_context=context,
                user_question=question
            )
        else:
            # Resposta geral sem contexto
            messages = self.general_response_template.format_messages(question=question)
        
        result = self.llm.invoke(messages)
        
        end_time = datetime.now()
        
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "agent": "gemini_rag",
            "response": response_text,
            "sources": sources or [],
            "num_sources": len(sources) if sources else 0,
            "timestamp": datetime.now().isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "model": self.model_name,
            "language": "pt-BR"
        }
    
    def generate_insights(
        self,
        data_summary: str,
        focus_area: Optional[str] = None
    ) -> Dict:
        """
        Gera insights a partir de dados
        
        Args:
            data_summary: Resumo dos dados
            focus_area: Área de foco para insights
        
        Returns:
            Dicionário com insights
        """
        prompt = f"""Analise os seguintes dados e forneça insights relevantes:

DADOS:
{data_summary}

{f'ÁREA DE FOCO: {focus_area}' if focus_area else ''}

Forneça insights acionáveis e relevantes em português."""
        
        result = self.general_chain.invoke({"question": prompt})
        
        return {
            "agent": "gemini_rag",
            "insights": result.get('text', ''),
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name
        }
    
    def set_rag_pipeline(self, rag_pipeline: MedicalRAGPipeline):
        """Define o pipeline RAG"""
        self.rag_pipeline = rag_pipeline
        print("✓ RAG pipeline configurado no Gemini Agent")


def main():
    """Teste do agente Gemini"""
    
    try:
        # Inicializar agente
        agent = GeminiRAGAgent()
        
        # Teste de resposta geral (sem RAG)
        result = agent.answer_general("O que é diabetes?")
        
        print("\n📊 RESULTADO DA RESPOSTA:")
        print(f"Agente: {result['agent']}")
        print(f"Tempo: {result['processing_time']:.2f}s")
        print(f"Resposta: {result['response']}")
        
        # Nota: Para testar com RAG, é necessário primeiro construir o vector store
        print("\n⚠️ Para usar RAG, execute primeiro o script de coleta de dados e construção do vector store")
        
    except Exception as e:
        print(f"⚠️ Erro ao testar agente: {e}")
        print("Certifique-se de que GOOGLE_API_KEY está configurado no .env")


if __name__ == "__main__":
    main()
