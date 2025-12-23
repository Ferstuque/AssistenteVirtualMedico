"""
Exports dos agentes médicos
"""

from src.agents.llama_agent import LlamaReasoningAgent as LlamaAgent
from src.agents.gemini_agent import GeminiRAGAgent as GeminiAgent

__all__ = ['LlamaAgent', 'GeminiAgent']
