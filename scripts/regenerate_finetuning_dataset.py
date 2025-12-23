"""
Script para regenerar datasets de fine-tuning limpos
Usa medquad_qa_pairs.json preservando terminologia médica
Gera arquivos compatíveis com Llama3 em UTF-8
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Configurações
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8  # 80% treino, 20% teste
MIN_ANSWER_LENGTH = 50  # Mínimo de caracteres para resposta válida

# System prompt em português correto (UTF-8)
SYSTEM_PROMPT = """Você é um assistente médico especializado em oncologia. Forneça análises clínicas estruturadas em 5 seções:
(1) Resumo da Condição
(2) Diagnósticos Diferenciais
(3) Investigações Recomendadas
(4) Nível de Urgência (EMERGÊNCIA/URGENTE/PRIORITÁRIO/ROTINA/CONSULTA)
(5) Recomendações ao Médico

NUNCA prescreva medicamentos diretamente. Sempre recomende que o médico responsável avalie e prescreva."""


def load_clean_data(file_path: Path) -> List[Dict]:
    """Carrega dados limpos do medquad_qa_pairs.json"""
    print(f"📂 Carregando dados de: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Carregados {len(data)} pares Q&A")
    return data


def validate_entry(entry: Dict) -> bool:
    """Valida se a entrada tem dados suficientes e não está corrompida"""
    # Verificar campos obrigatórios
    if not all(key in entry for key in ['question', 'answer']):
        return False
    
    # Verificar que não contém [PACIENTE] excessivo (corrupção)
    question = entry['question']
    answer = entry['answer']
    
    # Permitir no máximo 1 ocorrência de [PACIENTE] (alguns casos legítimos)
    if question.count('[PACIENTE]') > 1 or answer.count('[PACIENTE]') > 1:
        return False
    
    # Verificar tamanho mínimo da resposta
    if len(answer.strip()) < MIN_ANSWER_LENGTH:
        return False
    
    return True


def convert_to_llama3_format(entry: Dict) -> Dict:
    """Converte entrada para formato Llama3 de fine-tuning"""
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": entry['question']
            },
            {
                "role": "assistant",
                "content": entry['answer']
            }
        ],
        "metadata": {
            "source": entry.get('source', 'CancerGov'),
            "generated_at": datetime.now().isoformat(),
            "validated": True
        }
    }


def analyze_dataset(data: List[Dict], label: str) -> Dict:
    """Analisa dataset e retorna estatísticas"""
    stats = {
        'total_samples': len(data),
        'avg_question_length': sum(len(d['question']) for d in data) / len(data),
        'avg_answer_length': sum(len(d['answer']) for d in data) / len(data),
        'sources': {}
    }
    
    # Contar fontes
    for entry in data:
        source = entry.get('source', 'unknown')
        stats['sources'][source] = stats['sources'].get(source, 0) + 1
    
    # Verificar terminologia médica preservada (amostragem)
    medical_terms = [
        'lymphoblastic', 'leukemia', 'cancer', 'carcinoma', 'sarcoma',
        'chemotherapy', 'radiation', 'tumor', 'malignant', 'diagnosis'
    ]
    
    sample_text = ' '.join([d['question'] + ' ' + d['answer'] for d in data[:100]])
    sample_text_lower = sample_text.lower()
    
    stats['medical_terms_found'] = sum(1 for term in medical_terms if term in sample_text_lower)
    stats['has_corruption'] = '[PACIENTE]' in sample_text
    
    print(f"\n📊 Estatísticas - {label}")
    print(f"   Total de amostras: {stats['total_samples']}")
    print(f"   Tamanho médio pergunta: {stats['avg_question_length']:.0f} caracteres")
    print(f"   Tamanho médio resposta: {stats['avg_answer_length']:.0f} caracteres")
    print(f"   Termos médicos encontrados (amostra): {stats['medical_terms_found']}/{len(medical_terms)}")
    print(f"   Corrupção detectada: {'❌ SIM' if stats['has_corruption'] else '✅ NÃO'}")
    print(f"   Fontes: {stats['sources']}")
    
    return stats


def save_dataset(data: List[Dict], file_path: Path):
    """Salva dataset em JSON com encoding UTF-8"""
    print(f"\n💾 Salvando dataset em: {file_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    file_size_kb = file_path.stat().st_size / 1024
    print(f"✅ Salvo: {len(data)} amostras ({file_size_kb:.1f} KB)")


def main():
    """Função principal"""
    print("=" * 80)
    print("🔄 REGENERAÇÃO DE DATASETS DE FINE-TUNING")
    print("=" * 80)
    
    # Definir caminhos
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    input_file = data_dir / 'processed' / 'medquad_qa_pairs.json'
    output_dir = data_dir / 'finetuning'
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados limpos
    raw_data = load_clean_data(input_file)
    
    # Validar e filtrar dados
    print("\n🔍 Validando dados...")
    valid_data = [entry for entry in raw_data if validate_entry(entry)]
    
    invalid_count = len(raw_data) - len(valid_data)
    if invalid_count > 0:
        print(f"⚠️  Removidas {invalid_count} entradas inválidas ou corrompidas")
    
    print(f"✅ {len(valid_data)} entradas válidas para fine-tuning")
    
    # Converter para formato Llama3
    print("\n🔄 Convertendo para formato Llama3...")
    llama_data = [convert_to_llama3_format(entry) for entry in valid_data]
    
    # Embaralhar com seed fixo para reprodutibilidade
    random.seed(RANDOM_SEED)
    random.shuffle(llama_data)
    
    # Dividir em train/test
    split_idx = int(len(llama_data) * TRAIN_SPLIT)
    train_data = llama_data[:split_idx]
    test_data = llama_data[split_idx:]
    
    print(f"\n📊 Divisão dos dados:")
    print(f"   Treino: {len(train_data)} amostras ({TRAIN_SPLIT*100:.0f}%)")
    print(f"   Teste:  {len(test_data)} amostras ({(1-TRAIN_SPLIT)*100:.0f}%)")
    
    # Analisar datasets
    train_stats = analyze_dataset(
        [{'question': d['messages'][1]['content'], 
          'answer': d['messages'][2]['content'],
          'source': d['metadata']['source']} for d in train_data],
        "TREINO"
    )
    
    test_stats = analyze_dataset(
        [{'question': d['messages'][1]['content'], 
          'answer': d['messages'][2]['content'],
          'source': d['metadata']['source']} for d in test_data],
        "TESTE"
    )
    
    # Salvar datasets
    train_file = output_dir / 'train_llama3_clean.json'
    test_file = output_dir / 'test_llama3_clean.json'
    
    save_dataset(train_data, train_file)
    save_dataset(test_data, test_file)
    
    # Salvar metadados
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'source_file': str(input_file),
        'total_raw_samples': len(raw_data),
        'total_valid_samples': len(valid_data),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'train_split': TRAIN_SPLIT,
        'random_seed': RANDOM_SEED,
        'min_answer_length': MIN_ANSWER_LENGTH,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'encoding': 'utf-8',
        'format': 'llama3',
        'validation_passed': not (train_stats['has_corruption'] or test_stats['has_corruption'])
    }
    
    metadata_file = output_dir / 'dataset_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Metadados salvos em: {metadata_file}")
    
    # Validação final
    print("\n" + "=" * 80)
    print("✅ VALIDAÇÃO FINAL")
    print("=" * 80)
    
    validation_checks = [
        ("Encoding UTF-8", True),
        ("Formato Llama3", True),
        ("System prompt em português", SYSTEM_PROMPT.startswith("Você")),
        ("Sem corrupção de dados", metadata['validation_passed']),
        ("Termos médicos preservados", train_stats['medical_terms_found'] >= 8),
        ("Divisão train/test correta", len(train_data) + len(test_data) == len(valid_data))
    ]
    
    all_passed = True
    for check_name, passed in validation_checks:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 SUCESSO! Datasets regenerados e validados.")
        print(f"\n📁 Arquivos gerados:")
        print(f"   • {train_file.name} ({len(train_data)} amostras)")
        print(f"   • {test_file.name} ({len(test_data)} amostras)")
        print(f"   • {metadata_file.name} (metadados)")
    else:
        print("\n⚠️  ATENÇÃO: Algumas validações falharam. Revise os dados.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
