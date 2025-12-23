"""
Script de Demonstração - Análise de Prontuários de Pacientes
=============================================================
Demonstra como usar o Assistente Virtual Médico para análise
de prontuários e prognósticos de pacientes do hospital.
"""
import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.agents.gemini_agent import GeminiRAGAgent
from dotenv import load_dotenv


def print_separator(char="=", length=70):
    """Imprime uma linha separadora"""
    print(char * length)


def print_patient_info(patient_data):
    """Imprime informações resumidas do paciente"""
    print(f"\n👤 Paciente: {patient_data.get('patient_name', 'N/A')}")
    print(f"📋 ID: {patient_data.get('patient_id', 'N/A')}")
    print(f"🎂 Idade: {patient_data.get('patient_age', 'N/A')}")


def main():
    """Função principal de demonstração"""
    
    print_separator()
    print("DEMONSTRAÇÃO - ANÁLISE DE PRONTUÁRIOS DE PACIENTES")
    print_separator()
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Verificar API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ ERRO: GOOGLE_API_KEY não configurado no arquivo .env")
        print("   Configure sua API key antes de executar este script.")
        return
    
    try:
        # Inicializar agente Gemini
        print("\n🤖 Inicializando Gemini Agent...")
        agent = GeminiRAGAgent()
        
        print(f"\n✓ Agente inicializado com sucesso!")
        print(f"📊 Total de prontuários disponíveis: {len(agent.prontuarios)}")
        
        # Listar alguns pacientes
        print("\n" + "="*70)
        print("PACIENTES DISPONÍVEIS (Primeiros 5):")
        print("="*70)
        
        patients_list = agent.list_all_patients()
        for i, patient in enumerate(patients_list[:5], 1):
            print(f"\n{i}. ID: {patient['id_paciente']}")
            print(f"   Nome: {patient['nome']}")
            print(f"   Idade: {patient['idade']}")
        
        # Exemplo 1: Análise completa de paciente por ID
        print("\n" + "="*70)
        print("EXEMPLO 1: ANÁLISE COMPLETA DO PACIENTE José (ID: brcp230442)")
        print("="*70)
        
        result1 = agent.analyze_patient_prognosis(patient_id="brcp230442")
        
        if 'error' not in result1:
            print_patient_info(result1)
            print(f"\n⏱️  Tempo de processamento: {result1['processing_time']:.2f}s")
            print(f"\n📋 ANÁLISE CLÍNICA:")
            print("-" * 70)
            print(result1['response'])
        else:
            print(f"\n❌ Erro: {result1['error']}")
        
        # Exemplo 2: Análise com pergunta específica
        print("\n\n" + "="*70)
        print("EXEMPLO 2: PERGUNTA ESPECÍFICA SOBRE PACIENTE Maria")
        print("="*70)
        
        result2 = agent.analyze_patient_prognosis(
            patient_name="Maria",
            specific_question="Quais exames você recomendaria para confirmar o diagnóstico baseado nos sintomas apresentados?"
        )
        
        if 'error' not in result2:
            print_patient_info(result2)
            print(f"\n⏱️  Tempo de processamento: {result2['processing_time']:.2f}s")
            print(f"\n📋 RESPOSTA:")
            print("-" * 70)
            print(result2['response'])
        else:
            print(f"\n❌ Erro: {result2['error']}")
        
        # Exemplo 3: Análise de outro paciente
        print("\n\n" + "="*70)
        print("EXEMPLO 3: ANÁLISE DO PACIENTE Carlos (Diabetes)")
        print("="*70)
        
        result3 = agent.analyze_patient_prognosis(
            patient_id="brdb450123",
            specific_question="Avalie o risco cardiovascular deste paciente e sugira medidas preventivas."
        )
        
        if 'error' not in result3:
            print_patient_info(result3)
            print(f"\n⏱️  Tempo de processamento: {result3['processing_time']:.2f}s")
            print(f"\n📋 ANÁLISE DE RISCO:")
            print("-" * 70)
            print(result3['response'])
        else:
            print(f"\n❌ Erro: {result3['error']}")
        
        # Exemplo 4: Buscar paciente inexistente
        print("\n\n" + "="*70)
        print("EXEMPLO 4: TENTATIVA DE BUSCAR PACIENTE INEXISTENTE")
        print("="*70)
        
        result4 = agent.analyze_patient_prognosis(patient_id="XXXX99999")
        
        if 'error' in result4:
            print(f"\n⚠️  {result4['error']}")
            print(f"   Pacientes disponíveis no sistema: {result4['available_patients']}")
        
        # Resumo
        print("\n\n" + "="*70)
        print("RESUMO DA DEMONSTRAÇÃO")
        print("="*70)
        print(f"\n✓ Exemplos executados com sucesso!")
        print(f"✓ Total de prontuários disponíveis: {len(agent.prontuarios)}")
        print(f"✓ Modelo utilizado: {agent.model_name}")
        
        print("\n📚 COMO USAR NO SEU CÓDIGO:")
        print("-" * 70)
        print("""
# 1. Inicializar o agente
from src.agents.gemini_agent import GeminiRAGAgent
agent = GeminiRAGAgent()

# 2. Listar pacientes disponíveis
patients = agent.list_all_patients()

# 3. Buscar um paciente específico
patient = agent.search_patient(patient_id="brcp230442")
# ou
patient = agent.search_patient(patient_name="José")

# 4. Analisar prontuário e obter prognóstico
result = agent.analyze_patient_prognosis(
    patient_id="brcp230442",
    specific_question="Qual a probabilidade de câncer de pulmão?"
)

# 5. Acessar a resposta
print(result['response'])
print(result['patient_name'])
print(result['processing_time'])
        """)
        
        print("\n" + "="*70)
        print("DEMONSTRAÇÃO CONCLUÍDA")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
