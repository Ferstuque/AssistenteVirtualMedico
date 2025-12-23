"""
Script de Coleta de Dados MedQuAD
==================================
Este script é usado para extrair dados brutos do dataset MedQuAD
e salvar em formato estruturado para processamento posterior.

USO:
    python scripts/data_collection.py

ATENÇÃO: Este script é fornecido apenas como apoio para futuras
         extrações de dados. Não execute a menos que seja necessário
         re-extrair os dados brutos.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict


class MedQuADDataCollector:
    """
    Coletor de dados para o dataset MedQuAD
    Extrai perguntas e respostas de arquivos XML
    """
    
    def __init__(self, raw_data_path: str = None):
        """
        Inicializa o coletor de dados
        
        Args:
            raw_data_path: Caminho para os dados brutos (XML)
        """
        if raw_data_path is None:
            # Caminho padrão relativo ao script
            script_dir = Path(__file__).parent
            self.raw_data_path = script_dir.parent / "data" / "raw"
        else:
            self.raw_data_path = Path(raw_data_path)
        
        print(f"📁 Caminho dos dados brutos: {self.raw_data_path}")
        
    def extract_qa_from_xml(self, xml_file: Path) -> List[Dict[str, str]]:
        """
        Extrai pares de perguntas e respostas de um arquivo XML
        
        Args:
            xml_file: Caminho para o arquivo XML
        
        Returns:
            Lista de dicionários com perguntas e respostas
        """
        qa_pairs = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Diferentes estruturas XML do MedQuAD
            # Formato 1: QAPairs
            for qa_pair in root.findall('.//QAPair'):
                question_elem = qa_pair.find('Question')
                answer_elem = qa_pair.find('Answer')
                
                if question_elem is not None and answer_elem is not None:
                    question = question_elem.text
                    answer = answer_elem.text
                    
                    if question and answer:
                        qa_pairs.append({
                            'question': question.strip(),
                            'answer': answer.strip(),
                            'source_file': xml_file.name,
                            'format': 'QAPair'
                        })
            
            # Formato 2: Question/Answer diretos
            questions = root.findall('.//Question')
            answers = root.findall('.//Answer')
            
            if len(questions) == len(answers) and len(questions) > 0:
                for q_elem, a_elem in zip(questions, answers):
                    if q_elem.text and a_elem.text:
                        # Evitar duplicatas do formato 1
                        qa_text = {
                            'question': q_elem.text.strip(),
                            'answer': a_elem.text.strip()
                        }
                        if not any(pair['question'] == qa_text['question'] for pair in qa_pairs):
                            qa_pairs.append({
                                **qa_text,
                                'source_file': xml_file.name,
                                'format': 'Direct'
                            })
            
        except Exception as e:
            print(f"⚠️  Erro ao processar {xml_file.name}: {str(e)}")
        
        return qa_pairs
    
    def collect_all_data(self) -> pd.DataFrame:
        """
        Coleta todos os dados dos arquivos XML
        
        Returns:
            DataFrame com todos os pares Q&A
        """
        print(f"\n🔍 Coletando dados de {self.raw_data_path}...")
        
        all_qa_pairs = []
        xml_files = list(self.raw_data_path.glob("*.xml"))
        
        if not xml_files:
            print(f"⚠️  Nenhum arquivo XML encontrado em {self.raw_data_path}")
            return pd.DataFrame()
        
        print(f"📄 Encontrados {len(xml_files)} arquivos XML")
        
        for xml_file in xml_files:
            qa_pairs = self.extract_qa_from_xml(xml_file)
            all_qa_pairs.extend(qa_pairs)
            
            if len(qa_pairs) > 0:
                print(f"   ✓ {xml_file.name}: {len(qa_pairs)} pares Q&A")
        
        if not all_qa_pairs:
            print("⚠️  Nenhum par Q&A foi extraído")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_qa_pairs)
        print(f"\n✓ Total coletado: {len(df)} pares de perguntas e respostas")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_dir: str = None):
        """
        Salva os dados coletados em múltiplos formatos
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório de saída (padrão: data/processed)
        """
        if df.empty:
            print("⚠️  Nenhum dado para salvar")
            return
        
        if output_dir is None:
            script_dir = Path(__file__).parent
            output_dir = script_dir.parent / "data" / "processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar como CSV
        csv_path = output_dir / "medquad_qa_pairs_extracted.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 CSV salvo: {csv_path}")
        
        # Salvar como JSON
        json_path = output_dir / "medquad_qa_pairs_extracted.json"
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        print(f"💾 JSON salvo: {json_path}")
        
        # Estatísticas
        print(f"\n📊 Estatísticas:")
        print(f"   Total de pares Q&A: {len(df)}")
        print(f"   Arquivos fonte únicos: {df['source_file'].nunique()}")
        print(f"   Tamanho médio da pergunta: {df['question'].str.len().mean():.0f} caracteres")
        print(f"   Tamanho médio da resposta: {df['answer'].str.len().mean():.0f} caracteres")


def main():
    """
    Função principal para executar a coleta de dados
    """
    print("=" * 70)
    print("COLETOR DE DADOS MEDQUAD")
    print("=" * 70)
    print("\n⚠️  ATENÇÃO: Este script re-extrai dados dos arquivos XML brutos.")
    print("   Use apenas se precisar re-coletar os dados originais.\n")
    
    # Confirmar execução
    resposta = input("Deseja continuar com a extração? (s/N): ").strip().lower()
    
    if resposta != 's':
        print("❌ Extração cancelada pelo usuário.")
        return
    
    # Executar coleta
    collector = MedQuADDataCollector()
    df = collector.collect_all_data()
    
    if not df.empty:
        collector.save_data(df)
        print("\n✓ Coleta de dados concluída com sucesso!")
    else:
        print("\n❌ Falha na coleta de dados.")


if __name__ == "__main__":
    main()
