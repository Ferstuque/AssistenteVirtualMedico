"""
Sistema de anonimização de dados médicos - Versão simplificada sem Presidio
"""
from typing import List, Dict, Tuple
import re
import hashlib
import random


class MedicalDataAnonymizer:
    """
    Anonimizador simplificado de dados médicos
    Usa regex para remover informações sensíveis básicas
    """
    
    def __init__(self):
        # Mapeamento de entidades anonimizadas (para consistência)
        self.entity_mapping = {}
        
        # Padrões regex para detecção de informações sensíveis
        self.patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'PATIENT_ID': r'\b(PAT|PATIENT|ID)[-_]?\d{4,10}\b',
            'MEDICAL_RECORD': r'\b(MR|PRONT|RECORD)[-_]?\d{4,10}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        }
    
    def anonymize_text(self, text: str, language: str = "en") -> Tuple[str, List[Dict]]:
        """
        Anonimiza texto usando regex simples
        
        Args:
            text: Texto a ser anonimizado
            language: Idioma do texto (não utilizado nesta versão)
        
        Returns:
            Tupla (texto_anonimizado, entidades_encontradas)
        """
        anonymized_text = text
        entities_found = []
        
        # Aplicar cada padrão
        replacements = {
            'EMAIL': '<EMAIL_ANONIMIZADO>',
            'PHONE': '<TELEFONE_ANONIMIZADO>',
            'PATIENT_ID': '<ID_ANONIMIZADO>',
            'MEDICAL_RECORD': '<PRONTUARIO_ANONIMIZADO>',
            'SSN': '<SSN_ANONIMIZADO>',
        }
        
        for entity_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, anonymized_text, re.IGNORECASE))
            for match in matches:
                entities_found.append({
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.85
                })
            
            anonymized_text = re.sub(
                pattern, 
                replacements[entity_type], 
                anonymized_text, 
                flags=re.IGNORECASE
            )
        
        return anonymized_text, entities_found
    
    def anonymize_dataset(self, data: List[Dict], text_fields: List[str] = None) -> List[Dict]:
        """
        Anonimiza um dataset completo
        
        Args:
            data: Lista de dicionários contendo dados
            text_fields: Campos de texto a serem anonimizados (default: ['question', 'answer'])
        
        Returns:
            Dataset anonimizado
        """
        if text_fields is None:
            text_fields = ['question', 'answer']
        
        anonymized_data = []
        total_entities = 0
        
        for item in data:
            anonymized_item = item.copy()
            item_entities = []
            
            for field in text_fields:
                if field in item and item[field]:
                    anonymized_text, entities = self.anonymize_text(item[field])
                    anonymized_item[field] = anonymized_text
                    item_entities.extend(entities)
                    total_entities += len(entities)
            
            # Adicionar metadados de anonimização
            anonymized_item['_anonymization_metadata'] = {
                'entities_found': len(item_entities),
                'entity_types': list(set(e['type'] for e in item_entities))
            }
            
            anonymized_data.append(anonymized_item)
        
        print(f"✓ Anonimizadas {total_entities} entidades em {len(data)} registros")
        
        return anonymized_data
    
    def create_sample_dataset(self, full_data: List[Dict], sample_size: int = 100) -> List[Dict]:
        """
        Cria dataset de exemplo anonimizado para o repositório
        
        Args:
            full_data: Dataset completo
            sample_size: Tamanho da amostra
        
        Returns:
            Dataset de exemplo anonimizado
        """
        # Selecionar amostra aleatória
        sample = random.sample(full_data, min(sample_size, len(full_data)))
        
        # Já retorna a amostra (os dados já foram anonimizados antes)
        return sample


def main():
    """Função de teste/demonstração"""
    anonymizer = MedicalDataAnonymizer()
    
    # Teste com exemplo
    test_text = """
    Patient John Doe (ID: PAT-12345) visited on 2024-01-15.
    Contact: john.doe@email.com, phone: 555-1234.
    Address: 123 Main St, New York.
    Medical Record: MR-98765
    """
    
    anonymized, entities = anonymizer.anonymize_text(test_text)
    
    print("Original:")
    print(test_text)
    print("\nAnonimizado:")
    print(anonymized)
    print(f"\nEntidades encontradas: {len(entities)}")
    for entity in entities:
        print(f"  - {entity['type']} (score: {entity['score']:.2f})")


if __name__ == "__main__":
    main()
