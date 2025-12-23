"""
Script de Fine-Tuning do Llama 3 com LoRA
Adaptado para treinar com dados médicos usando GPU
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalFineTuner:
    """
    Fine-tuner para modelos médicos usando LoRA
    """
    
    def __init__(
        self,
        base_model: str,
        config: Dict,
        output_dir: str
    ):
        self.base_model = base_model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Dispositivo: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.2f} GB VRAM)")
        else:
            logger.warning("⚠️  GPU não disponível - treinamento será MUITO lento!")
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Carregar modelo base com LoRA"""
        logger.info(f"📦 Carregando modelo: {self.base_model}")
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Configurar padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Carregar modelo base
        model_kwargs = {
            "torch_dtype": torch.float16 if self.config.get("fp16", False) else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs
        )
        
        # Preparar para LoRA
        if self.config.get("use_lora", True):
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configurar LoRA
            lora_config = LoraConfig(
                r=self.config.get("lora_r", 16),
                lora_alpha=self.config.get("lora_alpha", 32),
                target_modules=self.config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=self.config.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # Imprimir parâmetros treináveis
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"📊 Parâmetros treináveis: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
        
        logger.info("✅ Modelo carregado com sucesso!")
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Preparar dataset para treinamento"""
        logger.info(f"📚 Carregando dataset: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Formatar para training
        formatted_data = []
        for example in data:
            # Construir prompt completo
            messages = example.get("messages", [])
            if not messages:
                continue
            
            # Concatenar mensagens
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    text += f"<|system|>{content}<|end|>\n"
                elif role == "user":
                    text += f"<|user|>{content}<|end|>\n"
                elif role == "assistant":
                    text += f"<|assistant|>{content}<|end|>\n"
            
            formatted_data.append({"text": text})
        
        logger.info(f"✅ Dataset preparado: {len(formatted_data)} exemplos")
        
        # Converter para HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenizar
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.get("max_seq_length", 2048),
                padding="max_length",
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, train_data: str, eval_data: str = None):
        """Executar fine-tuning"""
        logger.info("🚀 Iniciando fine-tuning...")
        
        # Preparar datasets
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = self.prepare_dataset(eval_data) if eval_data else None
        
        # Configurar argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
            fp16=self.config.get("fp16", False) and torch.cuda.is_available(),
            bf16=self.config.get("bf16", False) and torch.cuda.is_available(),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_steps=self.config.get("eval_steps", 100),
            warmup_steps=self.config.get("warmup_steps", 100),
            weight_decay=self.config.get("weight_decay", 0.01),
            lr_scheduler_type=self.config.get("lr_scheduler", "cosine"),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            optim=self.config.get("optim", "paged_adamw_8bit"),
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",  # Desabilitar wandb/tensorboard por padrão
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )
        
        # Criar Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Treinar
        logger.info("🏋️  Treinando modelo...")
        trainer.train()
        
        # Salvar modelo final
        logger.info("💾 Salvando modelo...")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        logger.info("✅ Fine-tuning concluído!")
        logger.info(f"📂 Modelo salvo em: {self.output_dir / 'final'}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 para dados médicos")
    parser.add_argument("--train_data", type=str, required=True, help="Caminho para dados de treino")
    parser.add_argument("--test_data", type=str, required=False, help="Caminho para dados de teste")
    parser.add_argument("--config", type=str, required=True, help="Caminho para configuração JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório de saída")
    
    args = parser.parse_args()
    
    # Carregar configuração
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    base_model = config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Criar fine-tuner
    fine_tuner = MedicalFineTuner(
        base_model=base_model,
        config=config,
        output_dir=args.output_dir
    )
    
    # Carregar modelo
    fine_tuner.load_model()
    
    # Treinar
    fine_tuner.train(
        train_data=args.train_data,
        eval_data=args.test_data
    )


if __name__ == "__main__":
    main()
