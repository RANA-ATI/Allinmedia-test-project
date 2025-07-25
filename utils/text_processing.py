"""
Text chunking and processing utilities
"""
import tqdm
import click
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter


class TextProcessor:
    """Handles text chunking operations"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def hybrid_split(self, pdf_texts, chunk_size=1000, chunk_overlap=0, tokens_per_chunk=256):
        """
        Split texts using hybrid approach (character + token splitting)
        
        Args:
            pdf_texts (list): List of PDF page texts
            chunk_size (int): Character chunk size
            chunk_overlap (int): Chunk overlap
            tokens_per_chunk (int): Tokens per chunk
            
        Returns:
            list: List of text chunks
        """
        if self.verbose:
            click.echo("Splitting text into character chunks...")
        
        # Character-based splitting
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        character_chunks = character_splitter.split_text('\n\n'.join(pdf_texts))
        
        if self.verbose:
            click.echo(f"Character chunks: {len(character_chunks)}")
            click.echo("Splitting into token chunks...")
        
        # Token-based splitting
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk
        )
        
        token_chunks = []
        iterator = tqdm.tqdm(character_chunks, desc="Processing chunks") if self.verbose else character_chunks
        
        for text in iterator:
            token_chunks.extend(token_splitter.split_text(text))
        
        if self.verbose:
            click.echo(f"Total token chunks: {len(token_chunks)}")
        
        return token_chunks
    
    def semantic_split(self, pdf_texts, model_name="BAAI/bge-base-en-v1.5", threshold_type="percentile"):
        """
        Split texts using semantic chunking
        
        Args:
            pdf_texts (list): List of PDF page texts
            model_name (str): Embedding model name
            threshold_type (str): Threshold type for semantic chunking
            
        Returns:
            list: List of semantic chunks (as strings)
        """
        if self.verbose:
            click.echo("Creating semantic chunks...")
        
        embed_model = FastEmbedEmbeddings(model_name=model_name)
        semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type=threshold_type)
        
        # Create documents using semantic chunker
        documents = semantic_chunker.create_documents(pdf_texts)
        
        # Extract text content
        chunks = [doc.page_content for doc in documents]
        
        if self.verbose:
            click.echo(f"Created {len(chunks)} semantic chunks")
        
        return chunks