"""
Document retrieval and reranking functionality
"""
import click
from sentence_transformers import CrossEncoder


class DocumentRetriever:
    """Handles document retrieval and reranking"""
    
    def __init__(self, vector_store, verbose=False):
        self.vector_store = vector_store
        self.verbose = verbose
    
    def retrieve_documents(self, query, n_results=10):
        """
        Retrieve documents from vector store
        
        Args:
            query (str): Search query
            n_results (int): Number of documents to retrieve
            
        Returns:
            list: Retrieved documents
        """
        if self.verbose:
            click.echo(f"Retrieving documents for query: '{query}'")
        
        results = self.vector_store.query(
            query_texts=query,
            n_results=n_results,
            include=['documents', 'embeddings']
        )
        
        documents = results['documents'][0]
        
        if self.verbose:
            click.echo(f"Retrieved {len(documents)} documents")
        
        return documents
    
    def rerank_documents(self, query, documents, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Rerank documents using cross-encoder
        
        Args:
            query (str): Search query
            documents (list): List of documents to rerank
            model_name (str): Cross-encoder model name
            
        Returns:
            list: Reranked documents
        """
        if self.verbose:
            click.echo("Re-ranking documents...")
        
        cross_encoder = CrossEncoder(model_name)
        pairs = [[query, doc] for doc in documents]
        scores = cross_encoder.predict(pairs)
        
        # Combine and sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs], [score for _, score in scored_docs]
    
    def retrieve_and_rerank(self, query, n_results=10, 
                           cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
                           show_rankings=False):
        """
        Complete retrieval and reranking pipeline
        
        Args:
            query (str): Search query
            n_results (int): Number of documents to retrieve
            cross_encoder_model (str): Cross-encoder model name
            show_rankings (bool): Whether to show document rankings
            
        Returns:
            list: Reranked documents
        """
        # Retrieve documents
        documents = self.retrieve_documents(query, n_results)
        
        # Rerank documents
        reranked_docs, scores = self.rerank_documents(query, documents, cross_encoder_model)
        
        # Show rankings if requested
        if show_rankings or self.verbose:
            click.echo("\nDocument Rankings:")
            for i, (doc, score) in enumerate(zip(reranked_docs, scores), 1):
                click.echo(f"Rank {i} | Score: {score:.4f}")
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                click.echo(preview)
                click.echo('-' * 50)
        
        return reranked_docs