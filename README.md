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
├── models/
│   └── llama3.1-8B-gptq/        # Quantized model after running notebook
├── utils/                       # Utility folder for general purpose functionalities
│   ├── __init__.py
│   ├── basic_utils.py
│   ├── embedding.py
│   ├── generator.py
│   ├── pipeline.py
│   ├── retriever.py
│   └── text_processing.py
├── main.py                      # Main application script
├── requirements.txt             # Requirements
└── README.md                    # This file
```

## Quantization Approaches

There are mainly 3 types of quantizations:
### 1. Post-Training Quantization (PTQ) – **Static**

**When:** After the model is fully trained.  
**How:**  
- Quantizes **weights** and **activations** to lower precision (e.g., INT8, INT4).  
- Uses a **calibration dataset** to compute fixed scale and zero-point values.  
- At inference, the same precomputed scales are used → **no runtime overhead**.

**Pros:**
- No retraining required.
- Efficient and fast inference.
- Works well for both weights and activations.

**Cons:**
- Needs a small calibration dataset.
- Can lose accuracy if quantization is too aggressive (especially below INT8).

**Examples:** GPTQ, AWQ, SmoothQuant, BRECQ.

---

### 2. Post-Training Quantization (PTQ) – **Dynamic**

**When:** After the model is fully trained.  
**How:**  
- Quantizes **weights** ahead of time.
- **Activations** are quantized dynamically at runtime, using scales computed for each batch.

**Pros:**
- No calibration dataset required.
- Easy to apply (e.g., PyTorch `quantize_dynamic`).
- Good for quick CPU speed-ups.

**Cons:**
- Usually limited to INT8.
- Slight runtime overhead from computing scales per batch.
- Not commonly used for INT4.

**Examples:** PyTorch dynamic quantization for Transformer-based NLP models.

---

### 3. Quantization-Aware Training (QAT)

**When:** During training or fine-tuning.  
**How:**  
- Simulates quantization effects during training (fake quantization).  
- Model learns to adapt to low-precision constraints.  
- Final trained model is converted to real quantized weights.

**Pros:**
- Best accuracy retention for very low bit-widths (e.g., INT4, INT2).
- Model is explicitly trained for quantization robustness.

**Cons:**
- Requires retraining or fine-tuning.
- Slower development cycle.

**Examples:** TensorFlow QAT, PyTorch QAT.

---

### 4. Hybrid / Mixed-Precision Quantization

**When:** Can be applied during PTQ or QAT.  
**How:**  
- Different layers use different precisions (e.g., sensitive layers in FP16, others in INT8 or INT4).  
- Balances speed, memory savings, and accuracy.

**Pros:**
- Flexibility to preserve accuracy in critical layers.
- Works well for hardware-optimized inference (e.g., NVIDIA AMP).

**Cons:**
- More complex to configure.
- May require manual tuning.

**Examples:** NVIDIA AMP (Automatic Mixed Precision), INT8+FP16 mixes.

---

### Summary Table

| Type | Applied | Weights | Activations | Calibration? | Retraining? | Common Bits |
|------|---------|---------|-------------|--------------|-------------|-------------|
| **PTQ - Static** | After training | Quantized | Quantized (fixed) | Yes | No | INT8, INT4 |
| **PTQ - Dynamic** | After training | Quantized | Quantized (per batch) | No | No | INT8 |
| **QAT** | During training | Quantized | Quantized | N/A | Yes | INT8, INT4, INT2 |
| **Hybrid** | Either | Mixed | Mixed | Sometimes | Sometimes | INT8+FP16, INT4+FP32 |

---

### Why GPTQ for INT4 Quantization?

I use **GPTQ (Gradient Post-Training Quantization)** for converting our model to INT4 because it is currently one of the **most accurate, widely adopted, and production-proven** post-training quantization techniques for Large Language Models.

**Key reasons for choosing GPTQ:**
- **Exceptional accuracy retention at low bit-widths**  
  GPTQ minimizes quantization error using a calibration dataset, keeping performance close to the original FP16/FP32 model even at 4-bit precision.
- **Low calibration data requirements**  
  Strong results can be achieved with as few as 128–512 calibration samples, making it feasible even without the original training data.
- **Optimized for fast inference**  
  Produces GPU-friendly quantized weights, enabling significant VRAM savings and high throughput with libraries like AutoGPTQ and ExLlama.
- **Mature ecosystem support**  
  Hugging Face, LMDeploy, and other frameworks have native GPTQ loaders, and pre-quantized GPTQ models are widely available.
- **Proven in real-world deployments**  
  Extensively tested on LLaMA, Falcon, MPT, and other LLMs, with strong benchmarks in both research and production environments.

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

## Required Hardware

This project uses **LLaMA 3.1**, quantized to **INT4** with GPTQ.  
Although INT4 greatly reduces VRAM usage, the model is still large and benefits from a modern GPU setup.

**Minimum Recommended:**
- **GPU:** NVIDIA RTX 3060 (12 GB VRAM) or higher  
  - For smoother performance on larger context sizes, 24 GB VRAM (RTX 3090/4090, A6000) is preferred.
- **CUDA:** 11.8+ (driver version compatible with your GPU)
- **RAM:** 16 GB system memory
- **CPU:** Modern multi-core CPU (for preprocessing, PDF parsing, and chunking)
- **Disk:** ~20 GB free storage for model and quantized weights


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