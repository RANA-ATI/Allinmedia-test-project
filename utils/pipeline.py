"""
Main RAG pipeline orchestration
"""
import click
from .basic_utils import load_pdf, save_response, print_header
from .text_processing import TextProcessor
from .embedding import VectorStore
from .retriever import DocumentRetriever
from .generator import ResponseGenerator


class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        
        # Initialize components
        self.text_processor = TextProcessor(verbose=verbose)
        self.vector_store = VectorStore(
            collection_name=config.get('collection_name', 'pdf_collection'),
            verbose=verbose
        )
        self.generator = ResponseGenerator(
            model_path=config.get('model_path', 'llama3.1-8B-gptq'),
            verbose=verbose
        )
        self.retriever = None  # Initialize after vector store is created
    
    def process_pdf(self, pdf_path):
        """Load and process PDF"""
        return load_pdf(pdf_path, verbose=self.verbose)
    
    def create_chunks(self, pdf_texts):
        """Create text chunks based on configuration"""
        chunking_method = self.config.get('chunking_method', 'semantic')
        
        if chunking_method == 'hybrid':
            return self.text_processor.hybrid_split(
                pdf_texts,
                chunk_size=self.config.get('chunk_size', 1000),
                chunk_overlap=self.config.get('chunk_overlap', 0),
                tokens_per_chunk=self.config.get('tokens_per_chunk', 256)
            )
        else:  # semantic
            return self.text_processor.semantic_split(
                pdf_texts,
                model_name=self.config.get('embedding_model', 'BAAI/bge-base-en-v1.5'),
                threshold_type=self.config.get('threshold_type', 'percentile')
            )
    
    def create_vector_store(self, chunks):
        """Create and populate vector store"""
        chunking_method = self.config.get('chunking_method', 'semantic')
        
        if chunking_method == 'hybrid':
            collection = self.vector_store.create_collection_with_default_embedding()
        else:  # semantic
            collection = self.vector_store.create_collection_with_custom_embedding(
                model_name=self.config.get('embedding_model', 'BAAI/bge-base-en-v1.5')
            )
        
        self.vector_store.add_documents(
            chunks, 
            batch_size=self.config.get('batch_size', 166)
        )
        
        # Initialize retriever
        self.retriever = DocumentRetriever(collection, verbose=self.verbose)
    
    def retrieve_documents(self, query):
        """Retrieve and rerank documents"""
        if not self.retriever:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        
        return self.retriever.retrieve_and_rerank(
            query,
            n_results=self.config.get('retrieval_count', 10),
            cross_encoder_model=self.config.get('cross_encoder_model', 
                                              'cross-encoder/ms-marco-MiniLM-L-6-v2'),
            show_rankings=self.config.get('show_rankings', False)
        )
    
    def generate_response(self, query, context_docs):
        """Generate final response"""
        return self.generator.generate_response(
            query,
            context_docs,
            max_new_tokens=self.config.get('max_new_tokens', 1024),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 0.95)
        )
    
    def run(self, pdf_path, query):
        """Run the complete RAG pipeline"""
        print_header(self.config)
        
        # Process PDF
        pdf_texts = self.process_pdf(pdf_path)
        
        # Create chunks
        chunks = self.create_chunks(pdf_texts)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        # Retrieve documents
        context_docs = self.retrieve_documents(query)
        
        # Generate response
        response = self.generate_response(query, context_docs)
        
        # Output results
        click.echo("\n" + "=" * 50)
        click.echo("FINAL ANSWER:")
        click.echo("=" * 50)
        click.echo(response)
        
        # Save response if specified
        output_file = self.config.get('output_file')
        if output_file:
            save_response(response, output_file)
        
        return response
