# main.py
"""
Main CLI interface using the modular RAG pipeline
"""
import click
from utils.pipeline import RAGPipeline

# Customizing how the click CLI handles help options (-h / --help).
### Explicitly enabling -h as a shortcut for help:
### python script.py -h will now work the same as: python script.py --help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# Defines the CLI entry point (function to run when invoked). E.g: python script.py	
@click.command(context_settings=CONTEXT_SETTINGS) # Yes required if you are using click as its entry point (for entry point)
# Positional parameter. Comes without a flag.
### type=click.Path(exists=True): Ensures the provided value is a valid path to a file or directory, and that it must exist on disk
@click.argument('pdf_path', type=click.Path(exists=True))
# Named flag or parameter. Often optional.
@click.option('--query', '-q', default="What are the stages benchmark supports?",
              help='Query to ask about the PDF') # --query or -q: These are the long and short flags for the option, "default": This sets the default value for the option if the user does not provide a --query or -q. "help": This is the description shown when the user runs: python script.py --help
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.option('--output-file', help='Save response to file')
@click.option('--show-rankings', is_flag=True,
              help='Show document rankings after retrieval')
# Click automatically sends the parameters to main function as arguments and kwargs
def main(pdf_path, query, **kwargs):
    """Process PDF with RAG pipeline using modular architecture"""
    # Create configuration dictionary
    config = {
        'pdf_path': pdf_path,
        'query': query,
        **kwargs # Note: If you hardcode the config dictionary, you’re repeating values already declared in @click.option(...). That means if you change or add a new CLI option (e.g., --top-k), you also have to update this dictionary manually.
    }
    
    # Initialize and run pipeline
    pipeline = RAGPipeline(config, verbose=kwargs.get('verbose', False))
    pipeline.run(pdf_path, query)

if __name__ == "__main__":
    main()