# Simple Local RAG Pipeline

> **Note**: This project is created for study and reference purposes, based on the excellent work by [Daniel Bourke](https://github.com/mrdbourke/simple-local-rag). Most of the implementation follows the original repository structure for educational learning.

## 🎯 Overview

Build a complete RAG (Retrieval Augmented Generation) pipeline from scratch that runs entirely locally on your machine. This project demonstrates how to create a "chat with PDF" application using open-source tools, focusing on privacy, speed, and cost-effectiveness.

### What We'll Build

- **Local RAG Pipeline**: Complete end-to-end workflow running on local NVIDIA GPU
- **PDF Chat Interface**: Query any PDF document using natural language
- **NutriChat Example**: Demonstrates querying a 1200-page nutrition textbook
- **Privacy-First**: All processing happens locally, no data sent to external APIs

## 🏗️ Architecture

```
PDF Input → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → LLM Generation → Response
```

### Key Components:
- **Document Processing**: PDF text extraction and intelligent chunking
- **Embedding Model**: Convert text to numerical representations
- **Vector Database**: Store and search document embeddings
- **Large Language Model**: Generate responses based on retrieved context
- **Query Interface**: Interactive chat interface for document queries

## 🚀 Features

- ✅ **100% Local Processing** - No external API calls required
- ✅ **Open Source Stack** - Built entirely with open-source tools
- ✅ **GPU Optimized** - Designed for NVIDIA GPU acceleration
- ✅ **Privacy Focused** - Your documents never leave your machine
- ✅ **Cost Effective** - No ongoing API costs after setup
- ✅ **Extensible** - Easy to customize for different document types

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 5GB+ VRAM (recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: 10GB+ free space for models

### Software Requirements
- **Python**: 3.11+ (tested on 3.11)
- **CUDA**: 12.1+ for GPU acceleration
- **Git**: For cloning the repository

### Knowledge Prerequisites
- Basic Python programming experience
- Familiarity with machine learning concepts
- Understanding of PyTorch basics (helpful but not required)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Surendra1341/simple-local-rag.git
cd simple-local-rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Activate Virtual Environment:**

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA Support

For optimal performance, install PyTorch with CUDA support:

```bash
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Note**: Adjust the CUDA version (cu121) based on your system setup. Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for specific instructions.

### 5. Setup Hugging Face Access

To use Gemma models, you need to:

1. **Accept Terms**: Visit [Gemma Model Page](https://huggingface.co/google/gemma-7b-it) and accept terms & conditions
2. **Authenticate**: Login via Hugging Face CLI
   ```bash
   huggingface-cli login
   ```

## 🏃‍♂️ Quick Start

### Option 1: Jupyter Notebook
```bash
jupyter notebook
```
Open and run `00-simple-local-rag.ipynb`

### Option 2: VS Code
```bash
code .
```

### Option 3: Python Script
```bash
python main.py
```

## 📊 What is RAG?

**RAG (Retrieval Augmented Generation)** combines the power of information retrieval with language generation:

### Core Components:
1. **Retrieval**: Find relevant information from your documents
2. **Augmentation**: Enhance LLM input with retrieved context  
3. **Generation**: Generate accurate, grounded responses

### Benefits:
- 🎯 **Reduces Hallucinations**: Provides factual context to LLMs
- 📚 **Custom Knowledge**: Works with your specific documents
- 🔍 **Source Attribution**: Know where answers come from
- ⚡ **Faster than Fine-tuning**: Quick implementation vs model retraining

## 🎯 Use Cases

### Business Applications
- **Customer Support**: Chat with documentation and FAQ databases
- **Internal Knowledge**: Query company policies and procedures  
- **Email Analysis**: Extract insights from email threads
- **Research Assistant**: Analyze technical papers and reports

### Educational Applications
- **Study Assistant**: Query textbooks and course materials
- **Research Papers**: Extract key information from academic literature
- **Documentation**: Navigate large technical documentation sets

## 🔧 Configuration

### Model Configuration
```python
# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Model  
LLM_MODEL = "google/gemma-7b-it"

# Vector Store Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5
```



### Performance Tips
- **GPU Memory**: Monitor VRAM usage with `nvidia-smi`
- **Chunk Size**: Experiment with different chunk sizes for your documents
- **Model Size**: Use smaller models if running out of memory

## 🤝 Contributing

This project is primarily for educational purposes. However, contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Learning Resources

- **Original Tutorial**: [Daniel Bourke's YouTube Walkthrough](https://youtu.be/qN_2fnOPY-M)
- **RAG Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **PyTorch Basics**: [Beginner PyTorch Video](https://youtu.be/Z_ikDlimN6A)

## 🙏 Acknowledgments

Special thanks to [Daniel Bourke (mrdbourke)](https://github.com/mrdbourke) for creating the original simple-local-rag project that serves as the foundation for this educational implementation.

## 📄 License

This project follows the same license as the original repository. Please check the original [simple-local-rag repository](https://github.com/mrdbourke/simple-local-rag) for license details.

## 🔗 Related Projects

- **Original Repository**: [mrdbourke/simple-local-rag](https://github.com/mrdbourke/simple-local-rag)
- **Nutrition Textbook**: [Hawaii OER - Human Nutrition](https://pressbooks.oer.hawaii.edu/humannutrition2/)

---

⭐ **Star this repository if you find it helpful for your learning journey!**

*Last updated: September 2025*
