"""
Testes para o sistema de métricas
"""
import pytest
from src.utils.metrics import MetricsEvaluator, ResponseQualityValidator


class TestMetrics:
    """Testes do sistema de métricas"""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture do evaluator"""
        return MetricsEvaluator()
    
    def test_bleu_calculation(self, evaluator):
        """Testa cálculo de BLEU"""
        reference = "This is a test sentence"
        hypothesis = "This is a test"
        
        scores = evaluator.calculate_bleu(reference, hypothesis)
        
        assert 'bleu_1' in scores
        assert 0 <= scores['bleu_1'] <= 1
    
    def test_rouge_calculation(self, evaluator):
        """Testa cálculo de ROUGE"""
        reference = "This is a test sentence"
        hypothesis = "This is a test"
        
        scores = evaluator.calculate_rouge(reference, hypothesis)
        
        assert 'rouge1_f1' in scores
        assert 0 <= scores['rouge1_f1'] <= 1
    
    def test_response_evaluation(self, evaluator):
        """Testa avaliação completa"""
        reference = "Diabetes is a chronic disease"
        generated = "Diabetes is a chronic condition"
        
        result = evaluator.evaluate_response(reference, generated)
        
        assert 'bleu_1' in result
        assert 'rouge1_f1' in result
        assert 'length_ratio' in result
    
    def test_medical_relevance(self):
        """Testa verificação de relevância médica"""
        validator = ResponseQualityValidator()
        
        response = "Diabetes affects blood sugar levels and insulin"
        keywords = ["diabetes", "blood", "sugar", "insulin"]
        
        score = validator.check_medical_relevance(response, keywords)
        assert score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
