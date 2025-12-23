"""
Script para limpar datasets e remover espaços excessivos.
Isso economiza tokens e melhora a performance do fine-tuning.
"""
import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove espaços excessivos preservando formatação importante."""
    if not text:
        return text
    
    # Remove múltiplos espaços em branco (mas preserva quebras de linha únicas)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove múltiplas quebras de linha (máximo 2 consecutivas)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove espaços antes/depois de quebras de linha
    text = re.sub(r' *\n *', '\n', text)
    
    # Remove tabs excessivos
    text = re.sub(r'\t+', ' ', text)
    
    # Strip geral
    text = text.strip()
    
    return text


def clean_dataset(input_path: Path, output_path: Path) -> dict:
    """Limpa um dataset JSON."""
    print(f"📖 Lendo: {input_path.name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = input_path.stat().st_size
    cleaned_count = 0
    
    # Limpar cada entrada
    for entry in data:
        if 'messages' in entry:
            for message in entry['messages']:
                if 'content' in message:
                    original = message['content']
                    cleaned = clean_text(original)
                    message['content'] = cleaned
                    if len(cleaned) < len(original):
                        cleaned_count += 1
    
    # Salvar dataset limpo
    print(f"💾 Salvando: {output_path.name}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    new_size = output_path.stat().st_size
    reduction_pct = ((original_size - new_size) / original_size) * 100
    
    stats = {
        'input_file': input_path.name,
        'output_file': output_path.name,
        'entries': len(data),
        'cleaned_messages': cleaned_count,
        'original_size_kb': original_size / 1024,
        'new_size_kb': new_size / 1024,
        'reduction_pct': reduction_pct
    }
    
    return stats


def main():
    """Processa os datasets de treinamento e teste."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'finetuning'
    
    print("🧹 Limpando datasets...\n")
    
    # Processar train
    train_input = data_dir / 'train_llama3_clean.json'
    train_output = data_dir / 'train_llama3_optimized.json'
    
    if train_input.exists():
        train_stats = clean_dataset(train_input, train_output)
        print(f"✓ Train: {train_stats['entries']} exemplos")
        print(f"  Tamanho: {train_stats['original_size_kb']:.1f} KB → {train_stats['new_size_kb']:.1f} KB")
        print(f"  Redução: {train_stats['reduction_pct']:.1f}%")
        print(f"  Mensagens limpas: {train_stats['cleaned_messages']}\n")
    
    # Processar test
    test_input = data_dir / 'test_llama3_clean.json'
    test_output = data_dir / 'test_llama3_optimized.json'
    
    if test_input.exists():
        test_stats = clean_dataset(test_input, test_output)
        print(f"✓ Test: {test_stats['entries']} exemplos")
        print(f"  Tamanho: {test_stats['original_size_kb']:.1f} KB → {test_stats['new_size_kb']:.1f} KB")
        print(f"  Redução: {test_stats['reduction_pct']:.1f}%")
        print(f"  Mensagens limpas: {test_stats['cleaned_messages']}\n")
    
    print("✅ Limpeza concluída!")
    print("\n📁 Arquivos otimizados:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")


if __name__ == '__main__':
    main()
