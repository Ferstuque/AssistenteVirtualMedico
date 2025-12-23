"""
Sistema de logging estruturado para o Assistente Virtual Médico
"""
import structlog
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import json


def setup_logging(log_level: str = "INFO", log_file: str = "logs/assistente_medico.log"):
    """
    Configura o sistema de logging estruturado
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho do arquivo de log
    """
    # Criar diretório de logs
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configurar structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configurar logging padrão do Python
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )
    
    # Adicionar handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level))
    logging.root.addHandler(file_handler)
    
    return structlog.get_logger()


class MedicalAssistantLogger:
    """Logger especializado para o assistente médico"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
    
    def log_request(self, user_query: str, session_id: str = None, metadata: Dict = None):
        """Registra uma requisição do usuário"""
        self.logger.info(
            "user_request",
            query=user_query,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )
    
    def log_agent_decision(self, agent_name: str, decision: str, reasoning: str = None, 
                          confidence: float = None, metadata: Dict = None):
        """Registra decisão de um agente"""
        self.logger.info(
            "agent_decision",
            agent=agent_name,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )
    
    def log_rag_retrieval(self, query: str, retrieved_chunks: list, 
                         num_chunks: int, sources: list, metadata: Dict = None):
        """Registra recuperação do RAG"""
        self.logger.info(
            "rag_retrieval",
            query=query,
            num_chunks=num_chunks,
            sources=sources,
            chunk_ids=[chunk.get('id') for chunk in retrieved_chunks if isinstance(chunk, dict)],
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )
    
    def log_response(self, response: str, response_time: float, 
                    agent_used: str, sources: list = None, metadata: Dict = None):
        """Registra resposta gerada"""
        self.logger.info(
            "response_generated",
            response=response[:200] + "..." if len(response) > 200 else response,
            response_time_seconds=response_time,
            agent=agent_used,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )
    
    def log_guardrail_violation(self, violation_type: str, input_text: str, 
                               reason: str, metadata: Dict = None):
        """Registra violação de guardrail"""
        self.logger.warning(
            "guardrail_violation",
            violation_type=violation_type,
            input=input_text[:100],
            reason=reason,
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )
    
    def log_error(self, error: Exception, context: str = None, metadata: Dict = None):
        """Registra erro"""
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            timestamp=datetime.now().isoformat(),
            **(metadata or {}),
            exc_info=True
        )
    
    def log_performance_metric(self, metric_name: str, metric_value: float, 
                              unit: str = None, metadata: Dict = None):
        """Registra métrica de performance"""
        self.logger.info(
            "performance_metric",
            metric=metric_name,
            value=metric_value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )
    
    def log_anonymization(self, original_entities: list, anonymized_count: int, 
                         metadata: Dict = None):
        """Registra processo de anonimização"""
        self.logger.info(
            "anonymization",
            entities_found=len(original_entities),
            entities_anonymized=anonymized_count,
            entity_types=[e.get('type') for e in original_entities if isinstance(e, dict)],
            timestamp=datetime.now().isoformat(),
            **(metadata or {})
        )


# Singleton instance
_logger_instance = None


def get_logger() -> MedicalAssistantLogger:
    """Retorna instância única do logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MedicalAssistantLogger()
    return _logger_instance
