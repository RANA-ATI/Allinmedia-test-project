import os
import sys
import tqdm
import torch
import chromadb
import argparse
import numpy as np
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process PDF with RAG pipeline')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('--query', '-q', default="What are the stages benchmark supports?", 
                       help='Query to ask about the PDF (default: "What are the stages benchmark supports?")')
    return parser.parse_args()


def load_and_process_pdf(pdf_path):
    """Load and extract text from PDF"""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        sys.exit(1)
    
    try:
        print(f"Loading PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        
        # Filter empty strings
        pdf_texts = [text for text in pdf_texts if text]
        
        if not pdf_texts:
            print("Error: No text found in the PDF file.")
            sys.exit(1)
            
        print(f"Successfully loaded {len(pdf_texts)} pages")
        print("First page preview:")
        print(pdf_texts[0][:500] + "..." if len(pdf_texts[0]) > 500 else pdf_texts[0])
        print("-" * 50)
        
        return pdf_texts
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)


# -------------- USE BELOW FUNCTION ONLY WHEN YOU WANT TO USE HYBRID APPROACH FOR CHUNKING --------------

# Utilizing Hybrid approach of langchain's text splitters to handle text chunking. 
### First I used recursive character text splitter to split the text into manageable chunks, then I used sentence transformers token text splitter to further
### split those chunks into token-based segments. Character Splitter is not enough due the reason that the embedder which we have to use has limited 256 characters or tokens context window

# def hybrid_split_texts(pdf_texts):
#     """Split texts into chunks with default parameters"""
#     print("Splitting text into character chunks...")
    
#     # Split the text using RecursiveCharacterTextSplitter
#     character_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ". ", " ", ""],
#         chunk_size=1000,
#         chunk_overlap=0
#     )
#     character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
#     print(f"Character chunks: {len(character_split_texts)}")

#     # Resplit the chunks using SentenceTransformersTokenTextSplitter
#     print("Splitting into token chunks...")
#     token_splitter = SentenceTransformersTokenTextSplitter(
#         chunk_overlap=0, 
#         tokens_per_chunk=256
#     )
#     token_split_texts = []
    
#     for text in tqdm.tqdm(character_split_texts, desc="Processing chunks"):
#         token_split_texts += token_splitter.split_text(text)

#     print(f"Total token chunks: {len(token_split_texts)}")
#     return token_split_texts


# def create_vector_store(token_split_texts):
#     """Create and populate ChromaDB collection"""
#     print("Creating embeddings and vector store...")
    
#     embedding_function = SentenceTransformerEmbeddingFunction()
    
#     # Create Chroma client and collection with unique name
#     chroma_client = chromadb.Client()
    
#     # Try to delete existing collection if it exists
#     try:
#         chroma_client.delete_collection("pdf_collection")
#     except:
#         pass
        
#     chroma_collection = chroma_client.create_collection(
#         "pdf_collection", 
#         embedding_function=embedding_function
#     )

#     # Prepare IDs and add documents in batches
#     ids = [str(i) for i in range(len(token_split_texts))]
#     batch_size = 166

#     print("Adding documents to vector store...")
#     for i in tqdm.tqdm(range(0, len(token_split_texts), batch_size), desc="Adding batches"):
#         batch_ids = ids[i:i + batch_size]
#         batch_texts = token_split_texts[i:i + batch_size]
#         chroma_collection.add(ids=batch_ids, documents=batch_texts)

#     print(f"Vector store created with {chroma_collection.count()} documents")
#     return chroma_collection


# -------------- USE BELOW FUNCTION ONLY WHEN YOU WANT TO USE SEMANTIC APPROACH FOR CHUNKING --------------

class ChromaCompatibleEmbedding:
    """Wrapper class to make FastEmbedEmbeddings compatible with ChromaDB"""
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.embed_model = FastEmbedEmbeddings(model_name=model_name)
    
    def __call__(self, input):
        """ChromaDB compatible embedding function"""
        if isinstance(input, str):
            input = [input]
        return self.embed_model.embed_documents(input)


model_name = "BAAI/bge-base-en-v1.5"
threshold_type = "percentile"

def semantic_split_texts_and_embedder(pdf_texts):
    # Create Chroma client and collection with unique name
    chroma_client = chromadb.Client()

    # Initialize the embedding model and chunker
    embed_model = FastEmbedEmbeddings(model_name=model_name)
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type=threshold_type)
    
    print("Creating semantic chunks...")
    # Create documents using the semantic chunker
    token_split_texts = semantic_chunker.create_documents(pdf_texts)
    print(f"Created {len(token_split_texts)} semantic chunks")

    # Extract the text content for each document
    documents = [doc.page_content for doc in token_split_texts]
    
    # Generate IDs for each chunk
    ids = [str(i) for i in range(len(documents))]

    # Try to delete existing collection if it exists
    try:
        chroma_client.delete_collection("allinmedia")
    except:
        pass
    
    # Create ChromaDB compatible embedding function
    chroma_embedding = ChromaCompatibleEmbedding(model_name=model_name)
        
    chroma_collection = chroma_client.create_collection(
        "allinmedia", 
        embedding_function=chroma_embedding
    )

    print("Adding documents to vector store...")
    # Store the documents in ChromaDB (embeddings will be generated automatically)
    batch_size = 166
    for i in tqdm.tqdm(range(0, len(documents), batch_size), desc="Adding batches"):
        batch_ids = ids[i:i + batch_size]
        batch_texts = documents[i:i + batch_size]
        chroma_collection.add(ids=batch_ids, documents=batch_texts)
    
    print(f"Vector store created with {chroma_collection.count()} documents")
    return chroma_collection


def retrieve_and_rerank(chroma_collection, query):
    """Retrieve documents and re-rank using cross-encoder with default parameters"""
    print(f"Retrieving documents for query: '{query}'")
    
    # Initial retrieval
    results = chroma_collection.query(
        query_texts=query, 
        n_results=10, 
        include=['documents', 'embeddings']
    )
    
    retrieved_documents = results['documents'][0]
    print(f"Retrieved {len(retrieved_documents)} documents")

    # Re-ranking with cross-encoder
    print("Re-ranking documents...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)

    # Combine and sort documents by score
    scored_docs = list(zip(retrieved_documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    print("\nDocument Rankings:")
    for i, (doc, score) in enumerate(scored_docs, 1):
        print(f"Rank {i} | Score: {score:.4f}")
        print(doc[:200] + "..." if len(doc) > 200 else doc)
        print('-' * 50)

    return [doc for doc, _ in scored_docs]


def generate_response(context_docs, query):
    """Generate response using the language model with local model"""
    model_path = "llama3.1-8B-gptq"
    print(f"Loading model from: {model_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model.eval()
        
        # Create context string
        context_string = "\n\n".join(context_docs)
        
        # Format prompt
        prompt_text = (
            "### System:\n"
            "You are a helpful expert research assistant. Your users are asking questions about information contained in the given data. "
            "You will be shown the user's question, and the relevant information from the user's data. "
            "Answer the user's question using only this information.\n\n"
            "### User:\n"
            f"Question: {query}\nInformation: {context_string}\n\n"
            "### Assistant:"
        )

        print("Generating response...")
        
        # Tokenize and generate
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "### Assistant:" in response:
            assistant_response = response.split("### Assistant:")[-1].strip()
        else:
            assistant_response = response
            
        return assistant_response
        
    except Exception as e:
        print(f"Error with model generation: {e}")
        return "Error: Could not generate response with the specified model."


def main():
    """Main function"""
    args = parse_arguments()
    
    print("=== PDF RAG Pipeline ===")
    print(f"PDF Path: {args.pdf_path}")
    print(f"Query: {args.query}")
    print("=" * 50)
    
    # Process PDF
    pdf_texts = load_and_process_pdf(args.pdf_path)
    
    # -------- ONLY UNCOMMENT BELOW WHEN YOU WANT TO USE HYBRID APPROACH FOR CHUNKING --------
    # # Split texts (using default parameters)
    # token_split_texts = hybrid_split_texts(pdf_texts)
    # 
    # # Create vector store
    # chroma_collection = create_vector_store(token_split_texts)
    
    # -------- ONLY UNCOMMENT BELOW WHEN YOU WANT TO USE SEMANTIC APPROACH FOR CHUNKING --------
    chroma_collection = semantic_split_texts_and_embedder(pdf_texts)

    # Retrieve and re-rank (using default parameters)
    context_docs = retrieve_and_rerank(chroma_collection, args.query)
    
    # Generate response (using default model)
    response = generate_response(context_docs, args.query)
    
    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(response)

if __name__ == "__main__":
    main()