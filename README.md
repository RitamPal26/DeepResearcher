# DeepResearcher 🔍

**An intelligent document analysis agent powered by RAG pipeline and CloudRift AI**

DeepResearcher transforms your documents into interactive knowledge bases. Upload PDFs or files, and get instant, contextual answers powered by advanced AI processing.

## 🏆 Hackathon Project

**Built for CodeMate Hackathon (September 20-21, 2025) organized by SRM IST and [CodeMate](https://edu.codemate.ai/) . Thank you for organizing the hackathon and giving me a chance to showcase my skills**

### Hackathon Requirements Met:
 - **Python-based system** for query handling and response generation  
 - **Local embedding generation** using Sentence Transformers for document indexing and retrieval  
 - **Multi-step reasoning** to break down queries into smaller tasks  
 - **Efficient storage and retrieval pipeline** using FAISS vector database  
 - **CodeMate Build integration** for project development and management  
 - **CodeMate Extension usage** for AI-powered coding assistance throughout development  

### CodeMate Tools Utilized:
- **Debug Code Agent**: Used for identifying and fixing code issues during development
- **Code Review Agent**: Applied for maintaining code quality and best practices
- **Generate Test Cases**: Automated test case generation for backend components
- **Chat Assistant**: Leveraged for code explanations and development guidance
- **Optimize Code Agent**: Enhanced code efficiency and performance


## ✨ Features

- **Document Upload**: Support for PDF and various file formats
- **RAG Pipeline**: Advanced chunking and processing using LangChain and FAISS
- **AI-Powered Analysis**: CloudRift Qwen model integration via OpenAI-compatible API
- **Vector Search**: FAISS-powered semantic search with sentence transformers
- **Interactive Queries**: Ask questions and get precise answers from your documents
- **Report Generation**: Comprehensive analysis and summary reports
- **Modern UI**: React-based interface with TypeScript and Markdown rendering

## 🛠️ Tech Stack

**Development Tools (Mandatory for Hackathon)**
- **CodeMate Build**: Web-based AI-powered development environment for project scaffolding and management
- **CodeMate VS Code Extension**: AI coding assistant with debugging, optimization, and code review capabilities

**Backend (Server)**
- **Framework**: FastAPI with Python 3.8+
- **RAG Pipeline**: LangChain for document processing and chaining
- **Vector Database**: FAISS for efficient similarity search
- **Embeddings**: Sentence Transformers for document encoding
- **PDF Processing**: PyPDF for document parsing
- **AI Integration**: OpenAI-compatible API for CloudRift Qwen model
- **Additional**: Pydantic for data validation, python-multipart for file uploads

**Frontend (Client)**
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for fast development and building
- **UI Components**: Modern React with Markdown rendering support
- **Styling**: Modern CSS with responsive design

## 🔧 Development Process

This project was built using **CodeMate's AI-powered development tools** as required by the hackathon:

### CodeMate Build
- Project initialization and scaffolding
- Integrated development environment for full-stack development
- AI-assisted project structure organization

### CodeMate VS Code Extension Features Used
- **AI Chat**: Natural language queries for coding assistance
- **Debug Agent**: Automated error detection and fixing
- **Code Review**: AI-powered code quality analysis
- **Test Generation**: Automated unit test case creation
- **Code Optimization**: Performance enhancement suggestions
- **Documentation**: Auto-generated docstrings and comments

*The mandatory use of these tools significantly accelerated the 24-hour development timeline while maintaining high code quality standards.*


## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ and npm
- CloudRift API key
- **CodeMate account** (for development workflow reproduction)
- **VS Code with CodeMate Extension** 

### Installation

1. **Clone the repository**
   ```
   git clone https://github.com/RitamPal26/DeepResearcher.git
   cd DeepResearcher
   ```

2. **Set up environment variables**
   ```
   # Copy example files
   cp .env.example .env
   ```

3. **Configure your API keys**
   
   **Edit `ai_env` file:**
   ```
   CLOUDRIFT_API_KEY=your_cloudrift_api_key_here
   ```

4. **Backend Setup**
   ```
   cd server
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start the FastAPI server
   uvicorn main:app --reload --port 8000
   ```

5. **Frontend Setup**
   ```
   cd client
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   ```

6. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## 📖 Usage

1. **Upload Documents**: Drag and drop PDF files or browse to select
2. **Processing**: The system automatically:
   - Extracts text using PyPDF
   - Chunks documents using LangChain text splitters
   - Creates embeddings with Sentence Transformers
   - Stores vectors in FAISS index
3. **Query**: Ask natural language questions about your documents
4. **AI Response**: Get intelligent answers via CloudRift Qwen model
5. **Results**: View responses with source citations and relevant context

## 🔧 Configuration

### Key Dependencies

**Python Backend:**
- `fastapi==0.116.2` - Modern web framework
- `langchain==0.3.27` - RAG pipeline orchestration
- `faiss-cpu==1.12.0` - Vector similarity search
- `sentence-transformers==5.1.0` - Document embeddings
- `pypdf==6.0.0` - PDF text extraction
- `openai==1.108.1` - CloudRift API integration

**React Frontend:**
- `react==19.1.1` - UI framework
- `typescript==5.8.3` - Type safety
- `vite==7.1.6` - Build tool
- `react-markdown==10.1.0` - Markdown rendering

## 🏗️ Architecture

```
DeepResearcher/
├── client/                 # React TypeScript frontend
│   ├── src/               # Source code
│   ├── package.json       # Node dependencies
│   └── vite.config.ts     # Vite configuration
├── server/                # FastAPI Python backend
│   ├── main.py           # FastAPI application
│   ├── requirements.txt  # Python dependencies
│   └── models/           # Data models
├── .env.example          # Environment template
└── ai_env.example        # AI services template
```

## 🎬 Demo

**[▶️ Watch Live Demo on Vimeo](https://vimeo.com/1120530120?share=copy)**

This video demonstrates the complete DeepResearcher workflow including document processing, AI-powered querying, and the RAG pipeline in action. Created as part of the CodeMate Hackathon requirements.


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Author

**Ritam Pal**
- GitHub: [@RitamPal26](https://github.com/RitamPal26)
- LinkedIn: [Ritam Pal](https://www.linkedin.com/in/ritam-pal-124175244/)
- Portfolio: [ritampal.dev](https://new-portfolio-lovat-one.vercel.app/)

## ⭐ Support

If you find this project helpful, please give it a star! ⭐

For issues and feature requests, please use the [Issues](https://github.com/RitamPal26/DeepResearcher/issues) tab.
