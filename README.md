# Allinmedia Test Project

A document question-answering system that processes PDF files and responds to queries using an optimized INT4 model. The system supports multiple text chunking approaches for optimal document processing.

## Prerequisites

- Python 3.7 or higher
- Required packages (see `requirements.txt`)

## Setup and Usage

### Step 1: Install Dependencies and Convert Model

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the notebooks folder and run the model conversion:
   ```bash
   cd notebooks
   jupyter notebook int4_conversion.ipynb
   ```
   
   **Note:** Run all cells in the notebook to convert and save the model. The converted model will be saved to your current working directory.

### Step 2: Run the Application

Execute the main script with your PDF file and query. The system supports different chunking approaches:

#### How to operate CLI
```bash
python main.py "path/to/your/document.pdf" --query "Your question here"
```

### Step 3: Get Results

The system will process your PDF document and return an answer to your query based on the document's content.

## Project Structure

```
Allinmedia-test-project/
├── notebooks/
│   └── int4_conversion.ipynb    # Model conversion notebook
├── data/
│   └── procyon_guide.pdf        # Sample PDF document
├── main.py                      # Main application script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Chunking Approaches

The system supports two different text chunking strategies:

### Hybrid Chunking (Recommended)
- Combines multiple chunking techniques for optimal performance
- Balances content preservation with processing efficiency
- Better handling of document structure and context

### Semantic Chunking (Alternative)
- Uses semantic similarity to create meaningful text chunks
- Preserves contextual relationships within the document
- May be slower but provides more coherent content groupings

**Default:** If no chunking method is specified, the system uses hybrid chunking.

## Usage Notes

- Ensure your PDF file path is correctly specified
- The system works best with text-based PDF documents
- Query responses are generated based on the content found in the provided PDF
- **Hybrid chunking** is recommended for most use cases as it provides better performance
- **Semantic chunking** can be used when document context preservation is critical
- If no chunking method is specified, the system defaults to hybrid chunking

- If you encounter import errors, verify all requirements are installed
- Make sure the model conversion step completed successfully before running queries
- Check that your PDF file path exists and is accessible