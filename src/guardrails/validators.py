"""
Guardrails usando Pydantic para validação e controle do assistente médico
"""
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional, Literal, Dict, Tuple
from enum import Enum
import re


class ResponseType(str, Enum):
    """Tipos de resposta permitidos"""
    INFORMATIONAL = "informational"  # Informação geral
    RECOMMENDATION = "recommendation"  # Recomendação ao médico
    CLARIFICATION = "clarification"  # Pedido de esclarecimento
    REFERRAL = "referral"  # Encaminhamento


class ConfidenceLevel(str, Enum):
    """Níveis de confiança da resposta"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MedicalResponse(BaseModel):
    """
    Modelo validado de resposta do assistente médico
    Implementa guardrails de segurança
    """
    
    response_text: str = Field(
        ..., 
        min_length=10,
        max_length=10000,
        description="Texto da resposta gerada"
    )
    
    response_type: ResponseType = Field(
        ...,
        description="Tipo de resposta"
    )
    
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Nível de confiança na resposta"
    )
    
    sources: List[str] = Field(
        default_factory=list,
        description="Fontes utilizadas (chunk IDs do RAG)"
    )
    
    contains_prescription: bool = Field(
        default=False,
        description="Flag indicando prescrição de medicamento"
    )
    
    contains_personal_info: bool = Field(
        default=False,
        description="Flag indicando informação pessoal identificável"
    )
    
    metadata: Optional[Dict] = Field(
        default_factory=dict,
        description="Metadados adicionais"
    )
    
    @field_validator('response_text')
    @classmethod
    def validate_no_direct_prescription(cls, v: str) -> str:
        """
        GUARDRAIL CRÍTICO: Proíbe prescrição direta de medicamentos
        """
        # Padrões que indicam prescrição direta
        prescription_patterns = [
            r'\btome\s+\d+\s*(mg|ml|comprimidos?|cápsulas?)',
            r'\bprescrevo\b',
            r'\bprescrição\s+de\b',
            r'\badministre\s+\d+',
            r'\btomar\s+\d+\s*vezes\s+ao\s+dia',
            r'\bdose\s+de\s+\d+',
        ]
        
        for pattern in prescription_patterns:
            if re.search(pattern, v.lower()):
                raise ValueError(
                    "GUARDRAIL VIOLATION: Prescrição direta de medicamentos não é permitida. "
                    "Use apenas recomendações ao médico responsável."
                )
        
        return v
    
    @field_validator('response_text')
    @classmethod
    def validate_no_personal_info(cls, v: str) -> str:
        """
        GUARDRAIL CRÍTICO: Proíbe vazamento de informações pessoais
        """
        # Padrões de informações pessoais
        pii_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Nomes próprios
            r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',  # CPF
            r'\b\d{2}/\d{2}/\d{4}\b',  # Datas completas
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # Email
            r'\b\d{4,5}-?\d{4}\b',  # Telefone
        ]
        
        # Verificações mais permissivas para contexto médico
        # Apenas bloqueia se houver múltiplos indicadores
        pii_count = sum(1 for pattern in pii_patterns if re.search(pattern, v))
        
        if pii_count >= 2:
            raise ValueError(
                "GUARDRAIL VIOLATION: Possível vazamento de informações pessoais. "
                "Todas as respostas devem ser completamente anonimizadas."
            )
        
        return v
    
    @field_validator('response_text')
    @classmethod
    def validate_professional_language(cls, v: str) -> str:
        """
        GUARDRAIL: Garante linguagem profissional e apropriada
        """
        # Palavras/frases inapropriadas
        inappropriate_terms = [
            'com certeza vai curar',
            'garantido',
            'milagroso',
            'cura definitiva',
        ]
        
        v_lower = v.lower()
        for term in inappropriate_terms:
            if term in v_lower:
                raise ValueError(
                    f"GUARDRAIL VIOLATION: Linguagem inapropriada detectada. "
                    f"Evite promessas absolutas ou termos não científicos."
                )
        
        return v
    
    def mark_as_prescription(self):
        """Marca resposta como contendo prescrição"""
        self.contains_prescription = True
    
    def mark_as_containing_pii(self):
        """Marca resposta como contendo informação pessoal"""
        self.contains_personal_info = True


class UserQuery(BaseModel):
    """Modelo validado de consulta do usuário"""
    
    query_text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Texto da consulta do usuário"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="ID da sessão"
    )
    
    language: str = Field(
        default="pt-BR",
        description="Idioma da consulta"
    )
    
    metadata: Optional[Dict] = Field(
        default_factory=dict,
        description="Metadados adicionais"
    )
    
    @field_validator('query_text')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Valida consulta do usuário"""
        # Remover espaços excessivos
        v = ' '.join(v.split())
        
        # Verificar se não está vazia após limpeza
        if len(v.strip()) < 3:
            raise ValueError("Consulta muito curta ou vazia")
        
        return v


class GuardrailValidator:
    """
    Validador centralizado de guardrails
    """
    
    @staticmethod
    def validate_response(response_data: Dict) -> MedicalResponse:
        """
        Valida resposta contra todos os guardrails
        
        Args:
            response_data: Dados da resposta
        
        Returns:
            MedicalResponse validado
        
        Raises:
            ValidationError: Se violação de guardrail
        """
        try:
            validated_response = MedicalResponse(**response_data)
            return validated_response
        except ValidationError as e:
            # Logar violação
            print(f"❌ GUARDRAIL VIOLATION: {e}")
            raise
    
    @staticmethod
    def validate_query(query_data: Dict) -> UserQuery:
        """
        Valida consulta do usuário
        
        Args:
            query_data: Dados da consulta
        
        Returns:
            UserQuery validado
        
        Raises:
            ValidationError: Se consulta inválida
        """
        try:
            validated_query = UserQuery(**query_data)
            return validated_query
        except ValidationError as e:
            print(f"❌ QUERY VALIDATION ERROR: {e}")
            raise
    
    @staticmethod
    def is_safe_response(response_text: str) -> Tuple[bool, List[str]]:
        """
        Verifica se resposta é segura (sem violar guardrails)
        
        Returns:
            Tupla (is_safe, violations)
        """
        violations = []
        
        try:
            MedicalResponse(
                response_text=response_text,
                response_type=ResponseType.INFORMATIONAL,
                confidence_level=ConfidenceLevel.MEDIUM
            )
            return True, []
        except ValidationError as e:
            for error in e.errors():
                violations.append(error['msg'])
            return False, violations


def main():
    """Teste dos guardrails"""
    validator = GuardrailValidator()
    
    # Teste 1: Resposta válida
    print("Teste 1: Resposta válida")
    try:
        valid_response = validator.validate_response({
            "response_text": "Recomendo que o médico avalie a possibilidade de investigar "
                           "os sintomas descritos com exames complementares.",
            "response_type": "recommendation",
            "confidence_level": "high",
            "sources": ["chunk_001", "chunk_023"]
        })
        print("✓ Resposta válida aceita")
    except ValidationError as e:
        print(f"✗ Erro: {e}")
    
    # Teste 2: Prescrição direta (DEVE FALHAR)
    print("\nTeste 2: Prescrição direta (deve falhar)")
    try:
        invalid_response = validator.validate_response({
            "response_text": "Tome 500mg de paracetamol 3 vezes ao dia.",
            "response_type": "recommendation",
            "confidence_level": "high"
        })
        print("✗ Prescrição direta foi aceita (ERRO!)")
    except ValidationError as e:
        print("✓ Prescrição direta bloqueada corretamente")
    
    # Teste 3: Validação de query
    print("\nTeste 3: Validação de query")
    try:
        valid_query = validator.validate_query({
            "query_text": "Quais são os sintomas de diabetes?",
            "session_id": "session_123"
        })
        print("✓ Query válida aceita")
    except ValidationError as e:
        print(f"✗ Erro: {e}")


if __name__ == "__main__":
    from typing import Tuple
    main()
