"""
Pipeline RAG (Retrieval-Augmented Generation) para assistente médico
Implementa chunking, embeddings e vector store
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np
from tqdm import tqdm


class MedicalRAGPipeline:
    """
    Pipeline completo de RAG para conhecimento médico
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vectorstore_path: str = "data/vectorstore"
    ):
        """
        Inicializa pipeline de RAG
        
        Args:
            embedding_model: Modelo de embeddings
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
            vectorstore_path: Caminho para salvar vector store
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar embeddings
        print("📊 Carregando modelo de embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter para chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vectorstore = None

    def _ensure_chunk_id(self, doc: Document, fallback_label: str) -> str:
        """Garante que todo chunk tenha um ID estável para explicabilidade."""
        existing_id = doc.metadata.get('chunk_id')
        if existing_id and existing_id != 'unknown':
            return existing_id

        source = str(doc.metadata.get('source', 'source')).replace(' ', '_')
        doc_id = doc.metadata.get('doc_id', 'no_doc')
        content_hash = hashlib.sha1(doc.page_content.encode('utf-8')).hexdigest()[:8]
        generated_id = f"chunk_{source}_{doc_id}_{fallback_label}_{content_hash}"
        doc.metadata['chunk_id'] = generated_id
        return generated_id
    
    def create_chunks(self, documents: List[Dict]) -> List[Document]:
        """
        Cria chunks dos documentos médicos
        
        Args:
            documents: Lista de dicionários com 'question', 'answer', 'source'
        
        Returns:
            Lista de Document objects
        """
        print(f"📝 Criando chunks de {len(documents)} documentos...")
        
        all_chunks = []
        chunk_id = 0
        
        for idx, doc in enumerate(tqdm(documents, desc="Processando documentos")):
            # Combinar pergunta e resposta para contexto completo
            combined_text = f"Pergunta: {doc.get('question', '')}\n\nResposta: {doc.get('answer', '')}"
            
            # Criar chunks
            chunks = self.text_splitter.create_documents(
                texts=[combined_text],
                metadatas=[{
                    'source': doc.get('source', 'unknown'),
                    'doc_id': idx,
                    'question': doc.get('question', ''),
                    'chunk_id': f"chunk_{chunk_id}"
                }]
            )
            
            # Atualizar chunk_ids
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"chunk_{chunk_id + i}"
            
            chunk_id += len(chunks)
            all_chunks.extend(chunks)
        
        print(f"✓ Criados {len(all_chunks)} chunks")
        print(f"  - Tamanho médio: {np.mean([len(c.page_content) for c in all_chunks]):.0f} caracteres")
        print(f"  - Min: {min(len(c.page_content) for c in all_chunks)}")
        print(f"  - Max: {max(len(c.page_content) for c in all_chunks)}")
        
        return all_chunks
    
    def build_vectorstore(self, chunks: List[Document], save: bool = True) -> FAISS:
        """
        Cria vector store a partir dos chunks
        
        Args:
            chunks: Lista de Document chunks
            save: Salvar vector store em disco
        
        Returns:
            FAISS vectorstore
        """
        print(f"🔢 Criando embeddings e vector store...")
        
        # Criar vectorstore
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print(f"✓ Vector store criado com {len(chunks)} chunks")
        
        # Salvar se solicitado
        if save:
            self.save_vectorstore()
        
        return self.vectorstore
    
    def save_vectorstore(self):
        """Salva vector store em disco"""
        if self.vectorstore is None:
            raise ValueError("Vector store não foi criado ainda")
        
        save_path = self.vectorstore_path / "faiss_index"
        self.vectorstore.save_local(str(save_path))
        print(f"✓ Vector store salvo em: {save_path}")
    
    def load_vectorstore(self) -> FAISS:
        """Carrega vector store do disco"""
        load_path = self.vectorstore_path / "faiss_index"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store não encontrado em {load_path}")
        
        print(f"📂 Carregando vector store de: {load_path}")
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Vector store carregado")
        
        return self.vectorstore
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Recupera documentos relevantes para a query
        
        Args:
            query: Texto da consulta
            k: Número de documentos a retornar
            score_threshold: Score mínimo de similaridade
        
        Returns:
            Lista de tuplas (Document, score)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store não carregado. Use load_vectorstore() ou build_vectorstore()")
        
        # Busca com scores
        results_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )
        
        # Filtrar por threshold
        filtered_results = [
            (doc, score) for doc, score in results_with_scores
            if score >= score_threshold
        ]
        
        return filtered_results
    
    def retrieve_with_metadata(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> Dict:
        """
        Recupera documentos com metadados completos para explicabilidade
        
        Args:
            query: Texto da consulta
            k: Número de documentos
            score_threshold: Score mínimo de distância FAISS
        
        Returns:
            Dicionário com resultados e metadados.
            Scores retornados são similaridades normalizadas [0, 1] onde 1 = mais similar.
        """
        results = self.retrieve(query, k, score_threshold)
        
        retrieval_info = {
            'query': query,
            'num_results': len(results),
            'chunks': []
        }
        
        for idx, (doc, score) in enumerate(results, start=1):
            chunk_id = self._ensure_chunk_id(doc, f"r{idx}")
            
            similarity = 1.0 - (float(score) / 2.0)
            similarity = max(0.0, min(1.0, similarity))  
            
            chunk_info = {
                'chunk_id': chunk_id,
                'content': doc.page_content,
                'score': similarity, 
                'source': doc.metadata.get('source', 'unknown'),
                'question': doc.metadata.get('question', ''),
                'doc_id': doc.metadata.get('doc_id', -1)
            }
            retrieval_info['chunks'].append(chunk_info)
        
        return retrieval_info
    
    def create_context_from_retrieval(self, retrieval_info: Dict, include_scores: bool = True) -> str:
        """
        Cria contexto formatado a partir dos resultados do RAG com explainability
        
        Args:
            retrieval_info: Informações de retrieval (scores já normalizados [0, 1])
            include_scores: Incluir scores de similaridade como porcentagem
        
        Returns:
            Contexto formatado com metadados de fonte.
            Scores são exibidos como porcentagem de similaridade (0-100%).
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieval_info['chunks'], 1):
            source_info = f"[Fonte {i} - ID: {chunk['chunk_id']} | Categoria: {chunk['source']}"
            if include_scores:
                source_info += f" | Relevância: {chunk['score']:.2%}"
            source_info += "]:"
            
            context_parts.append(
                f"{source_info}\n{chunk['content']}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def format_sources_summary(self, retrieval_info: Dict) -> str:
        """
        Formata um resumo das fontes utilizadas
        
        Args:
            retrieval_info: Informações de retrieval
        
        Returns:
            Resumo formatado das fontes
        """
        summary_parts = ["\n📚 FONTES UTILIZADAS:"]
        summary_parts.append("="*70)
        
        for i, chunk in enumerate(retrieval_info['chunks'], 1):
            summary_parts.append(
                f"\n{i}. Chunk ID: {chunk['chunk_id']}\n"
                f"   Categoria: {chunk['source']}\n"
                f"   Relevância: {chunk['score']:.2%}\n"
                f"   Trecho: {chunk['content'][:150]}..."
            )
        
        return "\n".join(summary_parts)


def build_rag_pipeline_from_data(data_file: str, output_dir: str = "data/vectorstore"):
    """
    Função helper para construir pipeline completo a partir de arquivo de dados
    
    Args:
        data_file: Caminho para arquivo JSON com dados
        output_dir: Diretório de saída
    """
    # Carregar dados
    print(f"📂 Carregando dados de: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Carregados {len(data)} documentos")
    
    # Criar pipeline
    rag = MedicalRAGPipeline(vectorstore_path=output_dir)
    
    # Criar chunks
    chunks = rag.create_chunks(data)
    
    # Construir vectorstore
    rag.build_vectorstore(chunks, save=True)
    
    print("\n✅ Pipeline RAG criado com sucesso!")
    
    return rag


def main():
    """Teste do pipeline RAG"""
    
    # Dados de exemplo para teste
    sample_data = [
        {
            "question": "What is cancer?",
            "answer": "Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells, which can invade nearby tissues and may metastasize to other parts of the body.",
            "source": "test"
        },
        {
            "question": "What are common signs of cancer?",
            "answer": "Warning signs can include persistent fatigue, unexplained weight loss, lumps or swelling, prolonged cough, abnormal bleeding, and changes in skin or moles, though specific symptoms depend on the cancer type.",
            "source": "test"
        }
    ]
    
    # Criar pipeline
    rag = MedicalRAGPipeline(vectorstore_path="data/vectorstore/test")
    
    # Processar
    chunks = rag.create_chunks(sample_data)
    rag.build_vectorstore(chunks, save=False)
    
    # Testar retrieval
    query = "What are diabetes symptoms?"
    results = rag.retrieve_with_metadata(query, k=2)
    
    print(f"\n🔍 Query: {query}")
    print(f"📊 Resultados: {results['num_results']}")
    for chunk in results['chunks']:
        print(f"\n  - Chunk ID: {chunk['chunk_id']}")
        print(f"    Score: {chunk['score']:.3f}")
        print(f"    Content: {chunk['content'][:100]}...")


if __name__ == "__main__":
    main()
