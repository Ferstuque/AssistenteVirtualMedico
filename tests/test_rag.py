"""
Testes para o pipeline RAG
"""
import pytest
from src.rag.pipeline import MedicalRAGPipeline


class TestRAGPipeline:
    """Testes do pipeline RAG"""
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo"""
        return [
            {
                "question": "What is diabetes?",
                "answer": "Diabetes is a chronic disease.",
                "source": "test"
            }
        ]
    
    def test_chunk_creation(self, sample_data):
        """Testa criação de chunks"""
        pipeline = MedicalRAGPipeline(chunk_size=100, chunk_overlap=20)
        chunks = pipeline.create_chunks(sample_data)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)
        assert all('chunk_id' in chunk.metadata for chunk in chunks)
    
    def test_context_creation(self):
        """Testa criação de contexto"""
        pipeline = MedicalRAGPipeline()
        
        retrieval_info = {
            'query': 'test',
            'num_results': 2,
            'chunks': [
                {
                    'chunk_id': 'chunk_001',
                    'content': 'Test content 1',
                    'score': 0.9
                },
                {
                    'chunk_id': 'chunk_002',
                    'content': 'Test content 2',
                    'score': 0.8
                }
            ]
        }
        
        context = pipeline.create_context_from_retrieval(retrieval_info)
        
        assert 'chunk_001' in context
        assert 'Test content 1' in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
