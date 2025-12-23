"""
Testes para o sistema de guardrails
"""
import pytest
from src.guardrails.validators import (
    GuardrailValidator,
    MedicalResponse,
    UserQuery,
    ResponseType,
    ConfidenceLevel
)
from pydantic import ValidationError


class TestGuardrails:
    """Testes de guardrails"""
    
    def test_valid_response(self):
        """Testa resposta válida"""
        response = MedicalResponse(
            response_text="Recomendo que o médico avalie os sintomas descritos.",
            response_type=ResponseType.RECOMMENDATION,
            confidence_level=ConfidenceLevel.HIGH,
            sources=["chunk_001"]
        )
        assert response.response_text is not None
    
    def test_prescription_blocking(self):
        """Testa bloqueio de prescrição direta"""
        with pytest.raises(ValidationError):
            MedicalResponse(
                response_text="Tome 500mg de paracetamol 3 vezes ao dia",
                response_type=ResponseType.RECOMMENDATION,
                confidence_level=ConfidenceLevel.HIGH
            )
    
    def test_valid_query(self):
        """Testa query válida"""
        query = UserQuery(
            query_text="O que é diabetes?"
        )
        assert query.query_text == "O que é diabetes?"
    
    def test_empty_query(self):
        """Testa query vazia"""
        with pytest.raises(ValidationError):
            UserQuery(query_text="  ")
    
    def test_validator_safe_response(self):
        """Testa verificação de segurança"""
        validator = GuardrailValidator()
        
        safe, violations = validator.is_safe_response(
            "Esta é uma resposta segura e apropriada."
        )
        assert safe is True
        assert len(violations) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
