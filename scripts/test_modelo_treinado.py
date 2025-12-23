"""
Script de teste para o modelo fine-tuned do Llama-3-70B.
Carrega o modelo treinado e testa com perguntas de oncologia.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
import torch


def load_finetuned_model(adapter_path: str):
    """Carrega o modelo base + adaptadores LoRA fine-tuned."""
    print(f"🔄 Carregando modelo fine-tuned de: {adapter_path}\n")
    
    # Modelo base (70B quantizado)
    base_model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    
    print("1️⃣ Carregando modelo base 70B...")
    
    # Configuração de quantização 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Carregar modelo base
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✓ Modelo base carregado")
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print("\n2️⃣ Carregando adaptadores LoRA fine-tuned...")
    
    # Carregar adaptadores LoRA
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("✓ Adaptadores LoRA carregados")
    print("✓ Modelo pronto para inferência!\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, user_question: str):
    """Gera resposta usando o modelo fine-tuned."""
    
    # Montar mensagens no formato de chat
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente médico especializado em oncologia. Forneça análises clínicas estruturadas em 5 seções:\n(1) Resumo da Condição\n(2) Diagnósticos Diferenciais\n(3) Investigações Recomendadas\n(4) Nível de Urgência (EMERGÊNCIA/URGENTE/PRIORITÁRIO/ROTINA/CONSULTA)\n(5) Recomendações ao Médico\n\nNUNCA prescreva medicamentos diretamente. Sempre recomende que o médico responsável avalie e prescreva."
        },
        {
            "role": "user",
            "content": user_question
        }
    ]
    
    # Aplicar template de chat
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gerar resposta com streaming
    print(f"💬 Pergunta: {user_question}\n")
    print("🤖 Resposta:")
    print("-" * 80)
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=768,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        use_cache=True
    )
    
    print("-" * 80)
    print()


def main():
    """Testa o modelo com várias perguntas."""
    
    # Path do modelo (ajuste se necessário)
    MODEL_PATH = "models/llama3_medical_ft"
    
    print("=" * 80)
    print("🏥 TESTE DO MODELO FINE-TUNED - ASSISTENTE MÉDICO DE ONCOLOGIA")
    print("=" * 80)
    print()
    
    # Carregar modelo
    model, tokenizer = load_finetuned_model(MODEL_PATH)
    
    # Perguntas de teste
    test_questions = [
        "O que é leucemia mieloide aguda?",
        "Quais são os sintomas de câncer de pulmão?",
        "Como é feito o diagnóstico de melanoma?",
    ]
    
    # Testar cada pergunta
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TESTE {i}/{len(test_questions)}")
        print(f"{'='*80}\n")
        
        generate_response(model, tokenizer, question)
        
        if i < len(test_questions):
            input("\nPressione ENTER para continuar...")
    
    print("\n" + "="*80)
    print("✅ TESTES CONCLUÍDOS!")
    print("="*80)


if __name__ == '__main__':
    main()
