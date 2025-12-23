"""
Script de Avaliação do Modelo Fine-Tunado
==========================================

Avalia o modelo Llama3 fine-tunado com métricas detalhadas:
- ROUGE (1, 2, L) - similaridade de n-gramas
- BLEU - qualidade de tradução/geração
- Perplexidade - confiança do modelo
- Exemplos de inferência interativa

Uso:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --interactive
    python scripts/evaluate_model.py --num_samples 10

Autor: Fernando Stuque Alves
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "llama3_medical_ft"
DATA_DIR = PROJECT_ROOT / "data" / "finetuning"
TEST_FILE = DATA_DIR / "test_llama3_clean.json"


def load_model_and_tokenizer(model_path: Path):
    """Carrega modelo fine-tunado e tokenizer."""
    logger.info(f"Carregando modelo de: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    logger.info(f"✓ Modelo carregado no device: {model.device}")
    return model, tokenizer


def load_test_data(test_file: Path) -> List[Dict]:
    """Carrega dataset de teste."""
    logger.info(f"Carregando test set: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"✓ {len(data)} exemplos carregados")
    return data


def format_prompt(system_msg: str, user_msg: str) -> str:
    """Formata prompt no formato Llama3."""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Gera resposta do modelo."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar apenas a parte gerada (remover prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calcula métricas ROUGE e BLEU."""
    logger.info("Calculando métricas...")
    
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    # ROUGE scores
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    # BLEU score
    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]  # BLEU espera lista de referências
    )
    
    metrics = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_score["bleu"]
    }
    
    logger.info("✓ Métricas calculadas")
    return metrics


def evaluate_on_test_set(model, tokenizer, test_data: List[Dict], num_samples: int = None):
    """Avalia modelo no test set completo."""
    if num_samples:
        test_data = test_data[:num_samples]
    
    logger.info(f"Avaliando em {len(test_data)} exemplos...")
    
    predictions = []
    references = []
    
    for example in tqdm(test_data, desc="Gerando predições"):
        messages = example["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")["content"]
        user_msg = next(m for m in messages if m["role"] == "user")["content"]
        reference = next(m for m in messages if m["role"] == "assistant")["content"]
        
        # Gerar predição
        prompt = format_prompt(system_msg, user_msg)
        prediction = generate_response(model, tokenizer, prompt)
        
        predictions.append(prediction)
        references.append(reference)
    
    # Calcular métricas
    metrics = calculate_metrics(predictions, references)
    
    # Exibir resultados
    print("\n" + "="*80)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*80)
    print(f"Exemplos avaliados: {len(test_data)}")
    print(f"\nMétricas:")
    print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"  BLEU:    {metrics['bleu']:.4f}")
    print("="*80)
    
    # Salvar resultados
    results_file = MODEL_DIR / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "num_samples": len(test_data),
            "metrics": metrics,
            "sample_predictions": [
                {
                    "question": test_data[i]["messages"][1]["content"],
                    "reference": references[i][:200] + "...",
                    "prediction": predictions[i][:200] + "..."
                }
                for i in range(min(3, len(test_data)))
            ]
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Resultados salvos em: {results_file}")
    
    return metrics, predictions, references


def interactive_mode(model, tokenizer):
    """Modo interativo para testar o modelo."""
    print("\n" + "="*80)
    print("MODO INTERATIVO - Assistente Médico Virtual")
    print("="*80)
    print("Digite perguntas médicas em inglês (ou 'sair' para terminar)")
    print("Exemplos:")
    print("  - What is (are) Breast Cancer?")
    print("  - What are the symptoms of Lung Cancer?")
    print("  - How to diagnose Colorectal Cancer?")
    print("-"*80)
    
    system_prompt = (
        "Você é um assistente médico especializado em oncologia. "
        "Forneça análises clínicas estruturadas em 5 seções:\n"
        "(1) Resumo da Condição\n"
        "(2) Diagnósticos Diferenciais\n"
        "(3) Investigações Recomendadas\n"
        "(4) Nível de Urgência (EMERGÊNCIA/URGENTE/PRIORITÁRIO/ROTINA/CONSULTA)\n"
        "(5) Recomendações ao Médico\n\n"
        "NUNCA prescreva medicamentos diretamente. Sempre recomende que o médico responsável avalie e prescreva."
    )
    
    while True:
        user_input = input("\n🩺 Pergunta: ").strip()
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando modo interativo...")
            break
        
        if not user_input:
            continue
        
        # Gerar resposta
        prompt = format_prompt(system_prompt, user_input)
        print("\n💭 Gerando resposta...\n")
        response = generate_response(model, tokenizer, prompt, max_tokens=1024)
        
        print("="*80)
        print("RESPOSTA:")
        print("="*80)
        print(response)
        print("="*80)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Avaliar modelo Llama3 fine-tunado")
    parser.add_argument("--num_samples", type=int, default=None, help="Número de exemplos a avaliar (padrão: todos)")
    parser.add_argument("--interactive", action="store_true", help="Modo interativo")
    parser.add_argument("--model_path", type=str, default=str(MODEL_DIR), help="Caminho do modelo")
    
    args = parser.parse_args()
    
    # Verificar se modelo existe
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"❌ Modelo não encontrado em: {model_path}")
        logger.info("Execute primeiro: python scripts/finetune_llama3.py")
        return
    
    # Carregar modelo
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    if args.interactive:
        # Modo interativo
        interactive_mode(model, tokenizer)
    else:
        # Avaliação no test set
        if not TEST_FILE.exists():
            logger.error(f"❌ Test set não encontrado: {TEST_FILE}")
            return
        
        test_data = load_test_data(TEST_FILE)
        evaluate_on_test_set(model, tokenizer, test_data, args.num_samples)


if __name__ == "__main__":
    main()
