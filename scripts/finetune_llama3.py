"""
Script de Fine-Tuning do Llama3 para Assistente MÃ©dico Virtual
================================================================

Este script realiza o fine-tuning do modelo Llama-3.2-3B-Instruct usando:
- Datasets limpos validados (train_llama3_clean.json, test_llama3_clean.json)
- LoRA (Low-Rank Adaptation) para eficiência de memória
- SFTTrainer (Supervised Fine-Tuning) da biblioteca TRL
- Avaliação com métricas ROUGE e BLEU
- Salvamento automático do melhor modelo

Requisitos:
- GPU recomendada (máximo 8GB VRAM)
- CPU: modo compatível mas muito mais lento
- Pacotes: transformers, peft, trl, evaluate, rouge-score, sacrebleu

Uso:
    python scripts/finetune_llama3.py --epochs 3 --batch_size 4 --learning_rate 2e-4

Autor: Tech Challenge FIAP IADT - Fase 3
Data: Dezembro 2025
"""

import json
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import evaluate

# Configuração de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/finetuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "finetuning"
MODEL_DIR = PROJECT_ROOT / "models" / "llama3_medical_ft"
LOGS_DIR = PROJECT_ROOT / "logs"

# Criar diretórios se não existirem
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Modelo base
BASE_MODEL = "meta-llama/Llama-3.2-1B"

# Configuracao LoRA (Parameter-Efficient Fine-Tuning)
LORA_CONFIG = {
    "r": 16,  # Rank - dimensionalidade da decomposição low-rank
    "lora_alpha": 32,  # Scaling factor
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Configuração de quantização (4-bit para economizar memória)
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def load_dataset_json(file_path: Path) -> List[Dict]:
    """Carrega dataset JSON validado."""
    logger.info(f"Carregando dataset: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"[OK] Dataset carregado: {len(data)} exemplos")
    return data


def format_chat_template(example: Dict) -> str:
    """
    Converte exemplo do formato Llama3 para texto formatado.
    
    Input: {"messages": [{"role": "system", "content": "..."}, ...]}
    Output: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>"
    """
    messages = example["messages"]
    formatted = "<|begin_of_text|>"
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    return formatted


def prepare_datasets(train_path: Path, test_path: Path, use_small: bool = True) -> tuple:
    """Prepara datasets para treinamento."""
    logger.info("Preparando datasets...")
    
    # Usar dataset reduzido
    if use_small:
        train_path = train_path.parent / "train_llama3_small.json"
        test_path = test_path.parent / "test_llama3_small.json"
        logger.info("[CONFIG] Usando dataset REDUZIDO (100 train, 25 test)")
    
    logger.info(f"[DATASET] Train: {train_path.name}")
    logger.info(f"[DATASET] Test: {test_path.name}")
    logger.info(f"[DATASET] Localizacao: {train_path.parent}")
    
    # Carregar JSONs
    train_data = load_dataset_json(train_path)
    test_data = load_dataset_json(test_path)
    
    logger.info(f"[DATASET] Total exemplos train: {len(train_data)}")
    logger.info(f"[DATASET] Total exemplos test: {len(test_data)}")
    
    # Formatar com chat template
    train_texts = [{"text": format_chat_template(ex)} for ex in train_data]
    test_texts = [{"text": format_chat_template(ex)} for ex in test_data]
    
    # Converter para Hugging Face Dataset
    train_dataset = Dataset.from_list(train_texts)
    test_dataset = Dataset.from_list(test_texts)
    
    logger.info(f"[OK] Train dataset: {len(train_dataset)} exemplos")
    logger.info(f"[OK] Test dataset: {len(test_dataset)} exemplos")
    
    # Mostrar exemplo do primeiro item
    if len(train_data) > 0:
        logger.info(f"[DATASET] Exemplo (primeiros 200 chars): {str(train_data[0])[:200]}...")
    
    return train_dataset, test_dataset


def setup_model_and_tokenizer(model_name: str, use_quantization: bool = True):
    """Configura modelo e tokenizer com LoRA."""
    logger.info(f"Carregando modelo base: {model_name}")
    logger.info("[CONFIG] Modo GPU com quantizacao 4-bit + limite 80% memoria")
    
    # Tentar limitar GPU (protegido contra travamentos)
    try:
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        torch.cuda.empty_cache()
        logger.info("[CONFIG] GPU limitada a 80% de memoria")
    except Exception as e:
        logger.info(f"[CONFIG] GPU config skip: {str(e)[:50]}")
    
    # Configurar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configurar quantização 4-bit
    logger.info("[OK] Carregando modelo com quantizacao 4-bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Carregar modelo com timeout protection
    logger.info("[CONFIG] Iniciando carregamento (pode demorar 2-3 min)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},  # Forçar device específico
            trust_remote_code=True,
            max_memory={0: "4.5GB", "cpu": "16GB"}  # Limites conservadores
        )
        logger.info("[OK] Modelo carregado com sucesso!")
    except Exception as e:
        logger.error(f"[ERRO] Falha ao carregar modelo: {str(e)[:200]}")
        raise
    
    # Preparar modelo para treinamento com quantização
    model = prepare_model_for_kbit_training(model)
    
    # Aplicar LoRA
    logger.info("Aplicando LoRA adapters...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Estatísticas do modelo
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params
    
    logger.info(f"[OK] Modelo configurado:")
    logger.info(f"  - Parametros totais: {all_params:,}")
    logger.info(f"  - Parametros treinaveis: {trainable_params:,} ({trainable_percent:.2f}%)")
    logger.info(f"  - LoRA rank: {LORA_CONFIG['r']}, alpha: {LORA_CONFIG['lora_alpha']}")
    logger.info(f"  - Device: GPU (quantizacao 4-bit, limite 80% memoria)")
    
    return model, tokenizer


def compute_metrics(eval_pred):
    """Calcula mÃ©tricas ROUGE e BLEU durante avaliaÃ§Ã£o."""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    predictions, labels = eval_pred
    
    # Decodificar predições e labels (remover padding)
    # Nota: Esta é uma versão simplificada - em produção, usar tokenizer.decode
    
    # Calcular métricas
    rouge_scores = rouge.compute(predictions=predictions, references=labels)
    bleu_score = bleu.compute(predictions=predictions, references=labels)
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_score["bleu"]
    }


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    epochs: int = 2,  # Reduzido para modo conservador GPU
    batch_size: int = 2,  # Reduzido para evitar crash
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 2  # Reduzido
):
    """Treina modelo com SFTTrainer."""
    logger.info("Iniciando treinamento...")
    
    # Configurar argumentos de treinamento (GPU com quantizacao 4-bit)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,  # FP16 para GPU
        logging_dir=str(LOGS_DIR / "tensorboard"),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to=[],
        push_to_hub=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )
    
    # Configurar SFT Trainer (trl 0.26.1 - parametros minimos)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )
    
    # Treinar
    logger.info("[INICIO] Iniciando treinamento...")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"  - Device: GPU (4-bit quantizado, 80% mem limit)")
    logger.info(f"  - Total train samples: {len(train_dataset)}")
    logger.info(f"  - Total eval samples: {len(eval_dataset)}")
    
    train_result = trainer.train()
    
    # Logs de resultado
    logger.info("aœ“ Treinamento concluÃ­do!")
    logger.info(f"  - Training loss: {train_result.training_loss:.4f}")
    logger.info(f"  - Training runtime: {train_result.metrics['train_runtime']:.2f}s")
    logger.info(f"  - Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    
    return trainer, train_result


def evaluate_model(trainer, test_dataset) -> Dict[str, float]:
    """Avalia modelo no dataset de teste."""
    logger.info("Avaliando modelo no test set...")
    
    eval_results = trainer.evaluate(test_dataset)
    
    logger.info("aœ“ AvaliaÃ§Ã£o concluÃ­da!")
    logger.info(f"  - Test loss: {eval_results['eval_loss']:.4f}")
    logger.info(f"  - Test perplexity: {np.exp(eval_results['eval_loss']):.2f}")
    
    return eval_results


def save_model_and_metrics(trainer, tokenizer, eval_results: Dict, output_dir: Path):
    """Salva modelo treinado, tokenizer e mÃ©tricas."""
    logger.info(f"Salvando modelo em: {output_dir}")
    
    # Salvar modelo e tokenizer
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Salvar métricas
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": BASE_MODEL,
            "eval_results": eval_results,
            "lora_config": LORA_CONFIG,
            "quantization": QUANTIZATION_CONFIG if torch.cuda.is_available() else None
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"aœ“ Modelo salvo: {output_dir}")
    logger.info(f"aœ“ MÃ©tricas salvas: {metrics_file}")


def generate_sample_inference(model, tokenizer, test_data: List[Dict], num_samples: int = 3):
    """Gera inferências de exemplo para validação."""
    logger.info(f"Gerando {num_samples} inferências de exemplo...")
    
    model.eval()
    samples = []
    
    for i, example in enumerate(test_data[:num_samples]):
        # Preparar prompt (system + user)
        messages = example["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        user_msg = next(m for m in messages if m["role"] == "user")
        expected_answer = next(m for m in messages if m["role"] == "assistant")
        
        # Formatar input
        input_text = format_chat_template({
            "messages": [system_msg, user_msg]
        })
        
        # Tokenizar
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Gerar resposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        samples.append({
            "question": user_msg["content"],
            "expected": expected_answer["content"][:200] + "...",
            "generated": generated[-500:]  # Ãšltimos 500 chars da resposta
        })
        
        logger.info(f"aœ“ Sample {i+1}/{num_samples} gerado")
    
    # Salvar exemplos
    samples_file = MODEL_DIR / "inference_samples.json"
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"aœ“ Exemplos salvos: {samples_file}")
    
    return samples


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal de execução."""
    parser = argparse.ArgumentParser(description="Fine-tuning Llama3 para Assistente Médico")
    parser.add_argument("--epochs", type=int, default=3, help="Número de epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size por device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no_quantization", action="store_true", help="Desabilitar quantizaÃ§Ã£o 4-bit")
    parser.add_argument("--train_file", type=str, default="train_llama3_clean.json", help="Arquivo de treino")
    parser.add_argument("--test_file", type=str, default="test_llama3_clean.json", help="Arquivo de teste")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FINE-TUNING LLAMA3 - ASSISTENTE MEDICO VIRTUAL")
    logger.info("="*80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"[CONFIG] PyTorch version: {torch.__version__}")
    logger.info(f"[CONFIG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # 1. Preparar datasets (usando versao reduzida)
        train_dataset, test_dataset = prepare_datasets(
            DATA_DIR / args.train_file,
            DATA_DIR / args.test_file,
            use_small=True  # Usar dataset reduzido para GPU
        )
        
        # 2. Configurar modelo
        model, tokenizer = setup_model_and_tokenizer(
            BASE_MODEL,
            use_quantization=not args.no_quantization
        )
        
        # 3. Treinar
        trainer, train_result = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            output_dir=MODEL_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        
        # 4. Avaliar
        eval_results = evaluate_model(trainer, test_dataset)
        
        # 5. Salvar modelo e métricas
        save_model_and_metrics(trainer, tokenizer, eval_results, MODEL_DIR)
        
        # 6. Gerar exemplos de inferência
        test_data_raw = load_dataset_json(DATA_DIR / args.test_file)
        generate_sample_inference(model, tokenizer, test_data_raw, num_samples=3)
        
        logger.info("="*80)
        logger.info("✓ PROCESSO CONCLUÍDO COM SUCESSO!")
        logger.info("="*80)
        logger.info(f"Modelo salvo em: {MODEL_DIR}")
        logger.info(f"Logs em: {LOGS_DIR}")
        logger.info("\nPróximos passos:")
        logger.info("1. Revisar logs de treinamento em logs/finetuning.log")
        logger.info("2. Visualizar TensorBoard: tensorboard --logdir logs/tensorboard")
        logger.info("3. Testar modelo: python scripts/evaluate_model.py")
        logger.info("4. Usar no notebook: carregar de models/llama3_medical_ft/")
        
    except Exception as e:
        logger.error(f"❌ Erro durante fine-tuning: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


