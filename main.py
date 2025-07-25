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