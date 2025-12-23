"""
Sistema de Avaliação de Acurácia e Métricas
Implementa múltiplas métricas para avaliar performance do assistente
"""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Imports opcionais para métricas avançadas
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge_score não disponível - métricas ROUGE desabilitadas")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  nltk não disponível - métricas BLEU desabilitadas")


class MetricsEvaluator:
    """
    Avaliador de métricas para o assistente médico
    """
    
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        if NLTK_AVAILABLE:
            self.smoothing = SmoothingFunction()
        else:
            self.smoothing = None
            
        self.evaluation_history = []
    
    def calculate_bleu(
        self,
        reference: str,
        hypothesis: str,
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Calcula BLEU score
        
        Args:
            reference: Texto de referência
            hypothesis: Texto gerado
            max_n: N-gramas máximos
        
        Returns:
            Dicionário com scores BLEU
        """
        if not NLTK_AVAILABLE:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        # Tokenizar
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        
        # Calcular BLEU
        weights = {
            1: (1.0, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (0.33, 0.33, 0.33, 0),
            4: (0.25, 0.25, 0.25, 0.25)
        }
        
        scores = {}
        for n in range(1, max_n + 1):
            score = sentence_bleu(
                [reference_tokens],
                hypothesis_tokens,
                weights=weights[n],
                smoothing_function=self.smoothing.method1
            )
            scores[f'bleu_{n}'] = score
        
        return scores
    
    def calculate_rouge(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict[str, float]:
        """
        Calcula ROUGE scores
        
        Args:
            reference: Texto de referência
            hypothesis: Texto gerado
        
        Returns:
            Dicionário com scores ROUGE
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {
                'rouge1_precision': 0.0, 'rouge1_recall': 0.0, 'rouge1_f1': 0.0,
                'rouge2_precision': 0.0, 'rouge2_recall': 0.0, 'rouge2_f1': 0.0,
                'rougeL_precision': 0.0, 'rougeL_recall': 0.0, 'rougeL_f1': 0.0,
            }
        
        scores = self.rouge_scorer.score(reference, hypothesis)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
    
    def calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[str] = None
    ) -> Dict[str, float]:
        """
        Calcula métricas de classificação
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            labels: Nomes das classes
        
        Returns:
            Dicionário com métricas
        """
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': np.mean(np.array(y_true) == np.array(y_pred))
        }
    
    def calculate_perplexity(
        self,
        log_probabilities: List[float]
    ) -> float:
        """
        Calcula perplexidade
        
        Args:
            log_probabilities: Log-probabilidades do modelo
        
        Returns:
            Perplexidade
        """
        if not log_probabilities:
            return float('inf')
        
        avg_log_prob = np.mean(log_probabilities)
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    def evaluate_response(
        self,
        reference: str,
        generated: str,
        metadata: Dict = None
    ) -> Dict:
        """
        Avalia uma resposta completa
        
        Args:
            reference: Resposta de referência
            generated: Resposta gerada
            metadata: Metadados adicionais
        
        Returns:
            Dicionário com todas as métricas
        """
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # BLEU
        bleu_scores = self.calculate_bleu(reference, generated)
        evaluation.update(bleu_scores)
        
        # ROUGE
        rouge_scores = self.calculate_rouge(reference, generated)
        evaluation.update(rouge_scores)
        
        # Métricas de comprimento
        evaluation['length_ratio'] = len(generated) / max(len(reference), 1)
        evaluation['reference_length'] = len(reference)
        evaluation['generated_length'] = len(generated)
        
        # Adicionar ao histórico
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def evaluate_batch(
        self,
        references: List[str],
        generated: List[str],
        metadata_list: List[Dict] = None
    ) -> Dict:
        """
        Avalia um lote de respostas
        
        Args:
            references: Lista de respostas de referência
            generated: Lista de respostas geradas
            metadata_list: Lista de metadados
        
        Returns:
            Dicionário com métricas agregadas
        """
        if len(references) != len(generated):
            raise ValueError("Número de referências e respostas geradas deve ser igual")
        
        metadata_list = metadata_list or [{}] * len(references)
        
        evaluations = []
        for ref, gen, meta in zip(references, generated, metadata_list):
            eval_result = self.evaluate_response(ref, gen, meta)
            evaluations.append(eval_result)
        
        # Agregar métricas
        aggregated = {
            'num_samples': len(evaluations),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calcular médias
        metric_keys = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
                      'rouge1_f1', 'rouge2_f1', 'rougeL_f1',
                      'length_ratio']
        
        for key in metric_keys:
            values = [e[key] for e in evaluations if key in e]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Gera relatório de avaliação
        
        Args:
            output_file: Arquivo de saída (opcional)
        
        Returns:
            Relatório formatado
        """
        if not self.evaluation_history:
            return "Nenhuma avaliação registrada"
        
        df = pd.DataFrame(self.evaluation_history)
        
        report_lines = [
            "=" * 80,
            "RELATÓRIO DE AVALIAÇÃO - ASSISTENTE MÉDICO VIRTUAL",
            "=" * 80,
            f"\nTotal de Avaliações: {len(self.evaluation_history)}",
            f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}",
            "\n" + "-" * 80,
            "MÉTRICAS MÉDIAS:",
            "-" * 80
        ]
        
        # Métricas principais
        metrics = {
            'BLEU-1': 'bleu_1',
            'BLEU-4': 'bleu_4',
            'ROUGE-1 F1': 'rouge1_f1',
            'ROUGE-L F1': 'rougeL_f1',
            'Length Ratio': 'length_ratio'
        }
        
        for name, key in metrics.items():
            if key in df.columns:
                mean = df[key].mean()
                std = df[key].std()
                report_lines.append(f"{name:20s}: {mean:.4f} (±{std:.4f})")
        
        report_lines.extend([
            "\n" + "-" * 80,
            "ESTATÍSTICAS DE COMPRIMENTO:",
            "-" * 80,
            f"Comprimento Médio Referência: {df['reference_length'].mean():.0f} caracteres",
            f"Comprimento Médio Gerado: {df['generated_length'].mean():.0f} caracteres",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        # Salvar se especificado
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Salvar também CSV
            csv_file = Path(output_file).with_suffix('.csv')
            df.to_csv(csv_file, index=False)
            
            print(f"✓ Relatório salvo em: {output_file}")
            print(f"✓ Dados CSV salvos em: {csv_file}")
        
        return report
    
    def save_evaluation_history(self, filepath: str):
        """Salva histórico de avaliações"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_history, f, ensure_ascii=False, indent=2)
        print(f"✓ Histórico de avaliações salvo em: {filepath}")
    
    def load_evaluation_history(self, filepath: str):
        """Carrega histórico de avaliações"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.evaluation_history = json.load(f)
        print(f"✓ Histórico de avaliações carregado de: {filepath}")


class ResponseQualityValidator:
    """
    Validador de qualidade das respostas
    """
    
    @staticmethod
    def check_medical_relevance(response: str, keywords: List[str]) -> float:
        """
        Verifica relevância médica da resposta
        
        Args:
            response: Texto da resposta
            keywords: Palavras-chave médicas esperadas
        
        Returns:
            Score de relevância (0-1)
        """
        response_lower = response.lower()
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        return matches / max(len(keywords), 1)
    
    @staticmethod
    def check_completeness(response: str, min_length: int = 50) -> bool:
        """Verifica se resposta é completa"""
        return len(response.strip()) >= min_length
    
    @staticmethod
    def check_source_citation(response: str) -> bool:
        """Verifica se resposta cita fontes"""
        patterns = ['[fonte:', '[chunk_', 'fonte:']
        return any(pattern in response.lower() for pattern in patterns)


def main():
    """Teste do sistema de avaliação"""
    evaluator = MetricsEvaluator()
    
    # Dados de teste
    reference = "Diabetes é uma doença crônica que afeta a forma como o corpo processa açúcar no sangue."
    generated = "Diabetes é uma condição médica crônica caracterizada por níveis elevados de glicose no sangue."
    
    # Avaliar
    result = evaluator.evaluate_response(reference, generated)
    
    print("📊 AVALIAÇÃO DE RESPOSTA:")
    print(f"BLEU-4: {result['bleu_4']:.4f}")
    print(f"ROUGE-1 F1: {result['rouge1_f1']:.4f}")
    print(f"ROUGE-L F1: {result['rougeL_f1']:.4f}")
    print(f"Length Ratio: {result['length_ratio']:.2f}")
    
    # Gerar relatório
    report = evaluator.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
