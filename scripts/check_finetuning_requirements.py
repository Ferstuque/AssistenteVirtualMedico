"""
Script de Verificação de Pré-Requisitos para Fine-Tuning
=========================================================

Verifica se o sistema está pronto para executar o fine-tuning:
- Hardware (GPU, RAM, espaço em disco)
- Software (Python, CUDA, dependências)
- Datasets (existem e estão válidos)
- Configurações

Uso:
    python scripts/check_finetuning_requirements.py

Autor: Tech Challenge FIAP IADT - Fase 3
"""

import sys
import json
import platform
from pathlib import Path

# Cores para terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_mark(passed: bool) -> str:
    """Retorna ✅ ou ❌ baseado no resultado."""
    return f"{Colors.GREEN}✅{Colors.END}" if passed else f"{Colors.RED}❌{Colors.END}"

def print_section(title: str):
    """Imprime cabeçalho de seção."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def check_python():
    """Verifica versão do Python."""
    print_section("1. Python")
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 10
    
    print(f"{check_mark(passed)} Python {version.major}.{version.minor}.{version.micro}")
    if not passed:
        print(f"{Colors.YELLOW}⚠️  Recomendado: Python 3.10 ou superior{Colors.END}")
    
    return passed

def check_gpu():
    """Verifica disponibilidade de GPU."""
    print_section("2. GPU/CUDA")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"{check_mark(True)} CUDA disponível")
            print(f"  • GPUs detectadas: {gpu_count}")
            print(f"  • Modelo: {gpu_name}")
            print(f"  • VRAM: {gpu_memory:.1f} GB")
            
            # Verificar VRAM mínima
            if gpu_memory < 8:
                print(f"{Colors.YELLOW}⚠️  VRAM baixa - considere batch_size=1 ou --no_quantization{Colors.END}")
            
            return True
        else:
            print(f"{check_mark(False)} GPU CUDA não disponível")
            print(f"{Colors.YELLOW}⚠️  Treinamento usará CPU (MUITO LENTO - 24-48h){Colors.END}")
            return False
            
    except ImportError:
        print(f"{check_mark(False)} PyTorch não instalado")
        print("Instale: pip install torch")
        return False

def check_ram():
    """Verifica RAM do sistema."""
    print_section("3. RAM")
    
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
        passed = ram_gb >= 16
        
        print(f"{check_mark(passed)} RAM: {ram_gb:.1f} GB")
        if not passed:
            print(f"{Colors.YELLOW}⚠️  Recomendado: mínimo 16GB RAM{Colors.END}")
        
        return passed
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  psutil não instalado - não foi possível verificar RAM{Colors.END}")
        return True  # Não bloquear se não conseguir verificar

def check_disk_space():
    """Verifica espaço em disco."""
    print_section("4. Espaço em Disco")
    
    try:
        import shutil
        project_root = Path(__file__).parent.parent
        stat = shutil.disk_usage(project_root)
        free_gb = stat.free / 1024**3
        passed = free_gb >= 15
        
        print(f"{check_mark(passed)} Espaço livre: {free_gb:.1f} GB")
        if not passed:
            print(f"{Colors.YELLOW}⚠️  Recomendado: mínimo 15GB livres{Colors.END}")
        
        return passed
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Não foi possível verificar espaço: {str(e)}{Colors.END}")
        return True

def check_dependencies():
    """Verifica dependências Python."""
    print_section("5. Dependências Python")
    
    required = {
        "transformers": "4.40.0",
        "peft": "0.10.0",
        "trl": "0.8.0",
        "accelerate": "0.28.0",
        "evaluate": "0.4.1",
        "datasets": "2.18.0"
    }
    
    optional = {
        "bitsandbytes": "0.43.0",  # GPU only
        "tensorboard": "2.15.0"
    }
    
    all_passed = True
    
    for package, min_version in required.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"{check_mark(True)} {package} ({version})")
        except ImportError:
            print(f"{check_mark(False)} {package} NÃO INSTALADO")
            all_passed = False
    
    print(f"\n{Colors.BOLD}Opcionais:{Colors.END}")
    for package, min_version in optional.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"{check_mark(True)} {package} ({version})")
        except ImportError:
            print(f"{Colors.YELLOW}⚠️  {package} não instalado (opcional){Colors.END}")
    
    if not all_passed:
        print(f"\n{Colors.YELLOW}Instale dependências faltantes:{Colors.END}")
        print("pip install -r requirements_finetuning.txt")
    
    return all_passed

def check_datasets():
    """Verifica existência e validade dos datasets."""
    print_section("6. Datasets")
    
    project_root = Path(__file__).parent.parent
    train_file = project_root / "data" / "finetuning" / "train_llama3_clean.json"
    test_file = project_root / "data" / "finetuning" / "test_llama3_clean.json"
    metadata_file = project_root / "data" / "finetuning" / "dataset_metadata.json"
    
    checks = [
        (train_file, "Train dataset"),
        (test_file, "Test dataset"),
        (metadata_file, "Metadata")
    ]
    
    all_passed = True
    
    for file_path, name in checks:
        exists = file_path.exists()
        print(f"{check_mark(exists)} {name}: {file_path.name}")
        
        if exists:
            size_mb = file_path.stat().st_size / 1024**2
            print(f"  • Tamanho: {size_mb:.2f} MB")
            
            # Validar JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    print(f"  • Exemplos: {len(data)}")
                elif isinstance(data, dict):
                    print(f"  • Campos: {len(data)}")
                    
            except json.JSONDecodeError:
                print(f"{Colors.RED}  • ERRO: JSON inválido{Colors.END}")
                all_passed = False
        else:
            all_passed = False
    
    if not all_passed:
        print(f"\n{Colors.YELLOW}Gere datasets limpos:{Colors.END}")
        print("python scripts/regenerate_finetuning_dataset.py")
    
    return all_passed

def check_model_dir():
    """Verifica se diretório de saída existe."""
    print_section("7. Diretório de Saída")
    
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "llama3_medical_ft"
    logs_dir = project_root / "logs"
    
    # Criar se não existir
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{check_mark(True)} models/llama3_medical_ft/ criado")
    print(f"{check_mark(True)} logs/ criado")
    
    return True

def main():
    """Executa todas as verificações."""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}VERIFICAÇÃO DE PRÉ-REQUISITOS - FINE-TUNING LLAMA3{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    
    results = {
        "Python": check_python(),
        "GPU/CUDA": check_gpu(),
        "RAM": check_ram(),
        "Disco": check_disk_space(),
        "Dependências": check_dependencies(),
        "Datasets": check_datasets(),
        "Diretórios": check_model_dir()
    }
    
    # Resumo final
    print_section("RESUMO")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for check, result in results.items():
        print(f"{check_mark(result)} {check}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} verificações aprovadas{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ SISTEMA PRONTO PARA FINE-TUNING!{Colors.END}")
        print(f"\n{Colors.BOLD}Execute:{Colors.END}")
        print("python scripts/finetune_llama3.py")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  CORRIJA OS PROBLEMAS ACIMA ANTES DE CONTINUAR{Colors.END}")
        print(f"\n{Colors.BOLD}Próximos passos:{Colors.END}")
        
        if not results["Dependências"]:
            print("1. Instalar dependências: pip install -r requirements_finetuning.txt")
        if not results["Datasets"]:
            print("2. Gerar datasets: python scripts/regenerate_finetuning_dataset.py")
        if not results["GPU/CUDA"]:
            print("3. Considere usar CPU (lento) ou configurar GPU CUDA")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
