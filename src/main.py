"""
Sistema Principal - Assistente Virtual Médico
Integra todos os componentes
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llama_agent import LlamaReasoningAgent
from src.agents.gemini_agent import GeminiRAGAgent
from src.rag.pipeline import MedicalRAGPipeline
from src.orchestrator import MedicalAssistantOrchestrator
from src.utils.logger import setup_logging, get_logger
from config.config import get_settings
import time


class MedicalAssistant:
    """
    Assistente Virtual Médico - Sistema Principal
    """
    
    def __init__(self):
        """Inicializa o assistente"""
        # Carregar configurações
        load_dotenv()
        self.settings = get_settings()
        
        # Setup logging
        setup_logging(
            log_level=self.settings.log_level,
            log_file=self.settings.log_file
        )
        self.logger = get_logger()
        
        print("🏥 ASSISTENTE VIRTUAL MÉDICO")
        print("=" * 80)
        
        # Inicializar componentes
        self._initialize_components()
        
        print("\n✅ Sistema inicializado e pronto!")
        print("=" * 80)
    
    def _initialize_components(self):
        """Inicializa todos os componentes do sistema"""
        
        # 1. RAG Pipeline
        print("\n📚 Carregando RAG Pipeline...")
        self.rag_pipeline = MedicalRAGPipeline(
            embedding_model=self.settings.embedding_model,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            vectorstore_path=self.settings.vectorstore_path
        )
        
        try:
            self.rag_pipeline.load_vectorstore()
        except FileNotFoundError:
            print("⚠️ Vector store não encontrado. Execute primeiro: python scripts/setup.py")
            sys.exit(1)
        
        # 2. Llama Agent
        print("\n🤖 Inicializando Llama Agent...")
        try:
            self.llama_agent = LlamaReasoningAgent(
                model_name=self.settings.llama_model_name,
                api_token=self.settings.huggingfacehub_api_token
            )
        except Exception as e:
            print(f"⚠️ Erro ao inicializar Llama Agent: {e}")
            print("   O sistema funcionará apenas com Gemini")
            self.llama_agent = None
        
        # 3. Gemini Agent
        print("\n🤖 Inicializando Gemini Agent...")
        self.gemini_agent = GeminiRAGAgent(
            model_name=self.settings.gemini_model_name,
            api_key=self.settings.google_api_key,
            rag_pipeline=self.rag_pipeline
        )
        
        # 4. Orchestrator
        print("\n🎭 Inicializando Orquestrador...")
        if self.llama_agent:
            self.orchestrator = MedicalAssistantOrchestrator(
                llama_agent=self.llama_agent,
                gemini_agent=self.gemini_agent,
                rag_pipeline=self.rag_pipeline
            )
        else:
            # Modo simplificado sem Llama
            self.orchestrator = None
    
    def ask(self, question: str) -> dict:
        """
        Faz uma pergunta ao assistente
        
        Args:
            question: Pergunta do usuário
        
        Returns:
            Resposta com metadados
        """
        start_time = time.time()
        
        if self.orchestrator:
            # Usar orquestrador completo
            result = self.orchestrator.process_query(question)
        else:
            # Usar apenas Gemini
            result = self.gemini_agent.answer_with_rag(question)
            result = {
                "response": result["response"],
                "sources": result["sources"],
                "confidence_level": "medium",
                "response_type": "informational",
                "urgency_level": "ROUTINE",
                "processing_steps": [],
                "guardrail_passed": True,
                "session_id": None
            }
        
        result["total_time"] = time.time() - start_time
        
        return result
    
    def interactive_mode(self):
        """Modo interativo de conversação"""
        print("\n💬 MODO INTERATIVO")
        print("Digite suas perguntas (ou 'sair' para encerrar)")
        print("-" * 80)
        
        while True:
            try:
                question = input("\n🔵 Você: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['sair', 'exit', 'quit']:
                    print("\n👋 Encerrando assistente...")
                    break
                
                print("\n⏳ Processando...")
                result = self.ask(question)
                
                print(f"\n🏥 Assistente: {result['response']}")
                print(f"\n📊 Metadados:")
                print(f"   - Fontes: {len(result['sources'])} chunks")
                print(f"   - Confiança: {result['confidence_level']}")
                print(f"   - Tipo: {result['response_type']}")
                print(f"   - Tempo: {result['total_time']:.2f}s")
                
                if result['sources']:
                    print(f"   - Sources: {', '.join(result['sources'][:3])}...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Encerrando assistente...")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                self.logger.log_error(e, context="interactive_mode")
    
    def batch_process(self, questions: list) -> list:
        """
        Processa lote de perguntas
        
        Args:
            questions: Lista de perguntas
        
        Returns:
            Lista de resultados
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\nProcessando {i}/{len(questions)}: {question[:50]}...")
            result = self.ask(question)
            results.append(result)
        
        return results


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assistente Virtual Médico")
    parser.add_argument(
        '--mode',
        choices=['interactive', 'single'],
        default='interactive',
        help='Modo de operação'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Pergunta única (modo single)'
    )
    
    args = parser.parse_args()
    
    # Inicializar assistente
    assistant = MedicalAssistant()
    
    if args.mode == 'interactive':
        assistant.interactive_mode()
    elif args.mode == 'single':
        if not args.question:
            print("❌ Erro: Especifique --question para modo single")
            return
        
        result = assistant.ask(args.question)
        print(f"\n🏥 Resposta:\n{result['response']}")
        print(f"\n📊 Fontes: {', '.join(result['sources'][:5])}")


if __name__ == "__main__":
    main()
