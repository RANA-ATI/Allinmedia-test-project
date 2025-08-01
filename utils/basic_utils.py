# utils.py
"""
Basic utility functions for PDF processing and common operations
"""
import os
import sys
import click
from pypdf import PdfReader


def load_pdf(pdf_path, verbose=False):
    """
    Load and extract text from PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        verbose (bool): Enable verbose output (For Debugging: Basically to show all the outputs those which are not meant for user to see.)
        
    Returns:
        list: List of text strings from each page
        
    Raises:
        SystemExit: If PDF file not found or no text extracted
    """
    if not os.path.exists(pdf_path):
        click.echo(click.style(f"Error: PDF file '{pdf_path}' not found.", fg='red'), err=True)
        sys.exit(1)
    
    try:
        if verbose:
            click.echo(f"Loading PDF: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        pdf_texts = [page.extract_text().strip() for page in reader.pages]
        
        # Filter empty strings
        pdf_texts = [text for text in pdf_texts if text]
        
        if not pdf_texts:
            click.echo(click.style("Error: No text found in the PDF file.", fg='red'), err=True)
            sys.exit(1)
            
        if verbose:
            click.echo(click.style(f"Successfully loaded {len(pdf_texts)} pages", fg='green'))
            preview_text = pdf_texts[0][:500] + "..." if len(pdf_texts[0]) > 500 else pdf_texts[0]
            click.echo(f"First page preview:\n{preview_text}")
            click.echo("-" * 50)
        
        return pdf_texts
        
    except Exception as e:
        click.echo(click.style(f"Error reading PDF: {e}", fg='red'), err=True)
        sys.exit(1)


def save_response(response, output_file):
    """
    Save response to file
    
    Args:
        response (str): Response text to save
        output_file (str): Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)
        click.echo(click.style(f"Response saved to: {output_file}", fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error saving response: {e}", fg='red'), err=True)


def print_header(config):
    """
    Print pipeline configuration header
    
    Args:
        config (dict): Configuration dictionary
    """
    click.echo("=== PDF RAG Pipeline ===")
    click.echo(f"PDF Path: {config.get('pdf_path', 'N/A')}")
    click.echo(f"Query: {config.get('query', 'N/A')}")
    click.echo(f"Chunking Method: {config.get('chunking_method', 'N/A')}")
    click.echo(f"Embedding Model: {config.get('embedding_model', 'N/A')}")
    click.echo(f"LLM Model: {config.get('model_path', 'N/A')}")
    click.echo("=" * 50)