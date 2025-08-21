# 3GPP Change Detection System

A sophisticated system for detecting, analyzing, and querying changes between different versions of 3GPP specifications. This tool parses DOCX documents, identifies structural and content changes, and provides an intelligent QA interface powered by vector search and large language models.

## 🚀 Features

### Core Capabilities
- **Document Parsing**: Extract and chunk 3GPP specification documents (DOCX format)
- **Change Detection**: Intelligent comparison between specification versions (e.g., Rel-15 vs Rel-17)
- **Vector Database**: Semantic search using FAISS with sentence transformers
- **QA Bot**: Natural language queries about specification changes
- **API Server**: RESTful API for programmatic access
- **Web Interface**: Interactive Streamlit dashboard for easy exploration

### Advanced Features
- **Hierarchical Processing**: Maintains document structure and section relationships
- **Smart Chunking**: Token-aware content segmentation with configurable limits
- **Change Tracking**: Detects additions, deletions, modifications, and moved content
- **Version Mapping**: Maps content between different specification versions
- **HTML Diff Generation**: Visual comparison of text changes
- **Event Clustering**: Groups related changes for better analysis

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for vector operations)
- ~2GB storage (for embeddings and processed data)

### Dependencies
See `requirements.txt` for complete list. Key dependencies include:
- **Document Processing**: python-docx, PyPDF2, beautifulsoup4
- **ML/NLP**: sentence-transformers, transformers, spacy
- **Vector DB**: faiss-cpu, chromadb
- **API/Web**: fastapi, streamlit, uvicorn
- **LLM Integration**: groq, langchain

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SwagatPatel03/Sasken_project.git
cd Sasken_project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# The sentence transformer model will be downloaded automatically on first use
```

### 5. Set Up Environment Variables
```bash
# Create .env file
cp .env.example .env  # If available, or create manually

# Add your API keys (optional, for Groq LLM)
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
```

## 📖 Usage

### Command Line Interface

The system provides a CLI with several commands:

#### 1. Parse Documents
```bash
cd src
python main.py parse --min_tokens 20 --max_tokens 500
```
This command:
- Parses DOCX files from `data/raw/`
- Creates hierarchical chunks
- Outputs JSON files to `data/processed/`

#### 2. Detect Changes
```bash
python main.py detect
```
This command:
- Maps chunks between old and new versions
- Detects all types of changes (ADD, DELETE, MODIFY, MOVE)
- Generates version mapping and change files

#### 3. Build Vector Database
```bash
python main.py builddb
```
This command:
- Creates FAISS embeddings for all chunks
- Builds event clusters from changes
- Stores vector indices for fast retrieval

#### 4. Start API Server
```bash
python main.py serve
```
Starts the FastAPI server on `http://localhost:8000`

### Web Interface

Start the Streamlit dashboard:
```bash
cd src
streamlit run app.py
```
Access the web interface at `http://localhost:8501`

### API Usage

#### Start the API Server
```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### Query the API
```python
import requests

# Ask a question about changes
response = requests.post(
    "http://localhost:8000/qa",
    json={
        "question": "What changed in section 5.5?",
        "top_k": 5
    }
)

answer = response.json()["answer"]
print(answer)
```

#### Example API Queries
```bash
# Using curl
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{"question": "Summarize authentication changes", "top_k": 10}'
```

## ⚙️ Configuration

Configuration is managed via `config/config.yaml`:

```yaml
# Project Configuration
project:
  name: "3GPP Change Detection System"
  version: "1.0.0"

# Model Configuration
models:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  llm_model: "llama-3.3-70b-versatile"

# Change Detection
change_detection:
  similarity_threshold: 0.75
  chunk_size: 512
  overlap: 20
  mapping_threshold: 0.6

# Vector Database
vector_db:
  type: "faiss"
  persist_directory: "data/embeddings"

# QA Bot
qa_bot:
  max_context_length: 12000
  temperature: 0.2
  top_k: 30
```

### Key Configuration Options

- **`similarity_threshold`**: Minimum similarity for change detection (0.0-1.0)
- **`chunk_size`**: Maximum tokens per document chunk
- **`mapping_threshold`**: Threshold for version mapping between chunks
- **`temperature`**: LLM creativity level (0.0 = deterministic, 1.0 = creative)

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Change        │    │   Vector        │
│   Parser        │───▶│   Detector      │───▶│   Database      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw DOCX      │    │   Changes.json  │    │   FAISS Index   │
│   Files         │    │   Version Map   │    │   Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                              ┌─────────────────────────────────────┐
                              │            QA Bot                   │
                              │  ┌─────────────┐ ┌─────────────────┐│
                              │  │   Vector    │ │      LLM        ││
                              │  │   Search    │ │   (Groq/Local)  ││
                              │  └─────────────┘ └─────────────────┘│
                              └─────────────────────────────────────┘
                                                       │
                              ┌─────────────────────────────────────┐
                              │        User Interfaces             │
                              │  ┌─────────────┐ ┌─────────────────┐│
                              │  │   CLI       │ │   Web App       ││
                              │  │   Tools     │ │   (Streamlit)   ││
                              │  └─────────────┘ └─────────────────┘│
                              │  ┌─────────────────────────────────┐│
                              │  │        REST API (FastAPI)      ││
                              │  └─────────────────────────────────┘│
                              └─────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**: DOCX files → Parsed chunks with metadata
2. **Change Detection**: Old chunks + New chunks → Change records
3. **Vector Processing**: Chunks → Embeddings → FAISS index
4. **Query Processing**: User question → Vector search → LLM generation → Answer

### Key Data Structures

#### Document Chunks
```json
{
  "section_id": "5.5.1.2",
  "title": "Attach procedure for EPS services",
  "content": "The attach procedure is used to...",
  "chunk_type": "content",
  "tokens": 245,
  "parent_section": "5.5.1"
}
```

#### Change Records
```json
{
  "change_type": "MODIFY",
  "old_chunk_id": "5.5.1.2_1",
  "new_chunk_id": "5.5.1.2_1",
  "similarity": 0.82,
  "description": "Content modified in section 5.5.1.2"
}
```

## 📚 API Documentation

### Endpoints

#### `POST /qa`
Ask questions about specification changes.

**Request:**
```json
{
  "question": "What authentication changes were made?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The main authentication changes include..."
}
```

#### Query Examples

**Basic Questions:**
- "What changed in section 5.5?"
- "Summarize all security modifications"
- "How many new procedures were added?"

**Advanced Queries:**
- "Compare authentication mechanisms between versions"
- "What are the implications of changes in bearer management?"
- "List all modifications to protocol configuration"

## 🧪 Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_change_detection.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Structure
```
src/
├── main.py              # CLI entry point
├── api.py               # FastAPI server
├── app.py               # Streamlit web interface
├── change_detection/    # Change detection algorithms
├── parsers/            # Document parsing utilities
├── qa_bot/             # QA system components
└── utils/              # Shared utilities

tests/                  # Test suite
config/                 # Configuration files
scripts/                # Utility scripts
```

### Adding New Features

1. **New Parser**: Extend `parsers/base_parser.py`
2. **Change Detection**: Modify `change_detection/detector.py`
3. **QA Enhancement**: Update `qa_bot/bot.py`
4. **API Endpoints**: Add to `api.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the src directory
cd src
python main.py --help
```

**Missing Models:**
```bash
# Download required models
python -m spacy download en_core_web_sm
```

**API Connection Issues:**
- Ensure the API server is running (`python main.py serve`)
- Check firewall settings for ports 8000 and 8501

### Getting Help

- Check the [Issues](https://github.com/SwagatPatel03/Sasken_project/issues) page
- Review test files for usage examples
- Examine the configuration file for customization options

## 🔮 Future Enhancements

- [ ] Support for PDF and HTML specifications
- [ ] Multi-language specification support
- [ ] Advanced visualization of changes
- [ ] Integration with 3GPP specification databases
- [ ] Automated report generation
- [ ] Batch processing capabilities
- [ ] Docker containerization
- [ ] Cloud deployment options

---

For more information about 3GPP specifications, visit the [official 3GPP website](https://www.3gpp.org/).