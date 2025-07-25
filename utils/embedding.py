"""
Embedding and vector store management
"""
import tqdm
import click
import chromadb
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaCompatibleEmbedding:
    """Wrapper class to make FastEmbedEmbeddings compatible with ChromaDB"""
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.embed_model = FastEmbedEmbeddings(model_name=model_name)
    
    def __call__(self, input):  # Changed from 'input_text' to 'input'
        """ChromaDB compatible embedding function"""
        if isinstance(input, str):
            input = [input]
        return self.embed_model.embed_documents(input)


class VectorStore:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, collection_name="pdf_collection", verbose=False):
        self.collection_name = collection_name
        self.verbose = verbose
        self.client = chromadb.Client()
        self.collection = None
    
    def create_collection_with_default_embedding(self):
        """Create collection with default SentenceTransformer embedding"""
        if self.verbose:
            click.echo("Creating vector store with default embeddings...")
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        embedding_function = SentenceTransformerEmbeddingFunction()
        self.collection = self.client.create_collection(
            self.collection_name,
            embedding_function=embedding_function
        )
        
        return self.collection
    
    def create_collection_with_custom_embedding(self, model_name="BAAI/bge-base-en-v1.5"):
        """Create collection with custom embedding model"""
        if self.verbose:
            click.echo(f"Creating vector store with custom embeddings: {model_name}")
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        embedding_function = ChromaCompatibleEmbedding(model_name=model_name)
        self.collection = self.client.create_collection(
            self.collection_name,
            embedding_function=embedding_function
        )
        
        return self.collection
    
    def add_documents(self, documents, batch_size=166):
        """
        Add documents to the vector store
        
        Args:
            documents (list): List of document texts
            batch_size (int): Batch size for adding documents
        """
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection_* first.")
        
        if self.verbose:
            click.echo("Adding documents to vector store...")
        
        # Generate IDs
        ids = [str(i) for i in range(len(documents))]
        
        # Add documents in batches
        iterator = tqdm.tqdm(
            range(0, len(documents), batch_size), 
            desc="Adding batches"
        ) if self.verbose else range(0, len(documents), batch_size)
        
        for i in iterator:
            batch_ids = ids[i:i + batch_size]
            batch_texts = documents[i:i + batch_size]
            self.collection.add(ids=batch_ids, documents=batch_texts)
        
        if self.verbose:
            click.echo(click.style(
                f"Vector store created with {self.collection.count()} documents", 
                fg='green'
            ))
    
    def get_collection(self):
        """Get the current collection"""
        return self.collection
