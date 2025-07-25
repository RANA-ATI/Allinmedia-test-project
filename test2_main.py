# main.py
"""
Main CLI interface using the modular RAG pipeline
"""
import click
from utils.pipeline import RAGPipeline

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--query', '-q', default="What are the stages benchmark supports?",
              help='Query to ask about the PDF')
@click.option('--chunking-method', type=click.Choice(['hybrid', 'semantic']), 
              default='semantic', help='Text chunking method')
@click.option('--chunk-size', type=int, default=1000,
              help='Character chunk size for hybrid method')
@click.option('--chunk-overlap', type=int, default=0,
              help='Chunk overlap for text splitting')
@click.option('--tokens-per-chunk', type=int, default=256,
              help='Tokens per chunk for hybrid method')
@click.option('--threshold-type', type=click.Choice(['percentile', 'standard_deviation', 'interquartile']),
              default='percentile', help='Semantic chunker threshold type')
@click.option('--embedding-model', default='BAAI/bge-base-en-v1.5',
              help='Embedding model name')
@click.option('--collection-name', default='pdf_collection',
              help='ChromaDB collection name')
@click.option('--retrieval-count', type=int, default=10,
              help='Number of documents to retrieve')
@click.option('--cross-encoder-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2',
              help='Cross-encoder model for reranking')
@click.option('--batch-size', type=int, default=166,
              help='Batch size for vector store operations')
@click.option('--model-path', default='llama3.1-8B-gptq',
              help='Path to the language model')
@click.option('--max-new-tokens', type=int, default=1024,
              help='Maximum new tokens to generate')
@click.option('--temperature', type=float, default=0.7,
              help='Generation temperature')
@click.option('--top-p', type=float, default=0.95,
              help='Top-p sampling parameter')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.option('--output-file', help='Save response to file')
@click.option('--show-rankings', is_flag=True,
              help='Show document rankings after retrieval')
def main(pdf_path, query, **kwargs):
    """Process PDF with RAG pipeline using modular architecture"""
    
    # Create configuration dictionary
    config = {
        'pdf_path': pdf_path,
        'query': query,
        **kwargs
    }
    
    # Initialize and run pipeline
    pipeline = RAGPipeline(config, verbose=kwargs.get('verbose', False))
    pipeline.run(pdf_path, query)


if __name__ == "__main__":
    main()