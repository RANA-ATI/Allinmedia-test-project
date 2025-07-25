"""
Response generation using language models
"""
import torch
import click
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResponseGenerator:
    """Handles response generation using language models"""
    
    def __init__(self, model_path = "E:/My Projects/Hegtavic Projects/All_in_media/Allinmedia-test-project/models/llama3.1-8B-gptq", verbose=False):
        self.model_path = "E:/My Projects/Hegtavic Projects/All_in_media/Allinmedia-test-project/models/llama3.1-8B-gptq"
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the language model and tokenizer"""
        if self.verbose:
            click.echo(f"Loading model from: {self.model_path}")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map="auto", 
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=True
            )
            self.model.eval()
            
            if self.verbose:
                click.echo(click.style("Model loaded successfully", fg='green'))
            
        except Exception as e:
            click.echo(click.style(f"Error loading model: {e}", fg='red'), err=True)
            raise
    
    def create_prompt(self, query, context_docs):
        """
        Create formatted prompt for the model
        
        Args:
            query (str): User query
            context_docs (list): List of context documents
            
        Returns:
            str: Formatted prompt
        """
        context_string = "\n\n".join(context_docs)
        
        prompt = (
            "### System:\n"
            "You are a helpful expert research assistant. Your users are asking questions "
            "about information contained in the given data. You will be shown the user's "
            "question, and the relevant information from the user's data. "
            "Answer the user's question using only this information.\n\n"
            "### User:\n"
            f"Question: {query}\nInformation: {context_string}\n\n"
            "### Assistant:"
        )
        
        return prompt
    
    def generate_response(self, query, context_docs, max_new_tokens=1024, 
                         temperature=0.7, top_p=0.95):
        """
        Generate response using the loaded model
        
        Args:
            query (str): User query
            context_docs (list): List of context documents
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Generation temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        if not self.model or not self.tokenizer:
            self.load_model()
        
        if self.verbose:
            click.echo("Generating response...")
        
        try:
            # Create prompt
            prompt_text = self.create_prompt(query, context_docs)
            
            # Tokenize
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "### Assistant:" in response:
                assistant_response = response.split("### Assistant:")[-1].strip()
            else:
                assistant_response = response
                
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error during generation: {e}"
            click.echo(click.style(error_msg, fg='red'), err=True)
            return "Error: Could not generate response with the specified model."