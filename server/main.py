import os
import uuid
import json
import tempfile
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Load API keys from your .env file
load_dotenv()

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Demo configuration - optimized for performance
DEMO_MODE = True
MAX_CACHE_SIZE = 100 if DEMO_MODE else 1000
DEFAULT_RETRIEVAL_K = 5 if DEMO_MODE else 7

# Enhanced document store with metadata and TTL
class DocumentStore:
    def __init__(self, vectorstore, metadata: dict, created_at: datetime):
        self.vectorstore = vectorstore
        self.metadata = metadata
        self.created_at = created_at
        self.access_count = 0
        self.last_accessed = created_at

document_stores: Dict[str, DocumentStore] = {}
query_cache: Dict[str, Any] = {}

# Background task executor - optimized for demo
executor = ThreadPoolExecutor(max_workers=4)

# Pre-warmed embeddings instance
_embeddings_instance = None

def get_optimized_embeddings():
    """Get optimized embeddings for faster demo performance"""
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise e
    return _embeddings_instance

# Pre-warm the model
async def pre_warm_model():
    """Pre-warm the embedding model to avoid cold start delays"""
    try:
        embeddings = get_optimized_embeddings()
        dummy_text = ["Warming up the model for demo"]
        await asyncio.get_event_loop().run_in_executor(
            executor, embeddings.embed_documents, dummy_text
        )
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

async def cleanup_old_documents():
    """Background task to clean up old document stores"""
    while True:
        try:
            current_time = datetime.now()
            expired_docs = []
            
            for doc_id, doc_store in document_stores.items():
                cleanup_threshold = timedelta(hours=2 if DEMO_MODE else 24)
                if (current_time - doc_store.created_at > cleanup_threshold and 
                    doc_store.access_count < 5):
                    expired_docs.append(doc_id)
            
            for doc_id in expired_docs:
                del document_stores[doc_id]
                logger.info(f"Cleaned up expired document: {doc_id}")
            
            if len(query_cache) > MAX_CACHE_SIZE:
                query_cache.clear()
                
            await asyncio.sleep(1800 if DEMO_MODE else 3600)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(1800 if DEMO_MODE else 3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Deep Researcher Agent API")
    # await pre_warm_model()
    cleanup_task = asyncio.create_task(cleanup_old_documents())
    yield
    # Shutdown
    cleanup_task.cancel()
    executor.shutdown(wait=True)
    logger.info("Deep Researcher Agent API shutdown complete")

# Initialize the FastAPI application with enhanced configuration
app = FastAPI(
    title="Deep Researcher Agent API",
    description="Advanced RAG system with multi-step reasoning and optimized retrieval",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
# Replace your current CORS configuration with this:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "http://localhost:3000",   # Alternative local port
        "https://deep-researcher-five.vercel.app",  # Production URL
        # Add specific preview URLs if needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global exception handler for debugging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {type(exc).__name__}: {str(exc)}",
            "type": type(exc).__name__
        }
    )

# Validation error handler
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )

# --- Enhanced Pydantic Models ---
class DocumentMetadata(BaseModel):
    filename: str
    file_size: int
    total_pages: int
    total_chunks: int
    processing_time: float
    upload_timestamp: datetime

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str
    metadata: DocumentMetadata

class QueryRequest(BaseModel):
    doc_id: str
    question: str
    max_reasoning_steps: Optional[int] = Field(default=2, ge=1, le=5)
    enable_multi_step: Optional[bool] = Field(default=True)

class ReasoningStep(BaseModel):
    step_number: int
    sub_question: str
    retrieved_sources: List[int]
    partial_answer: str
    confidence_score: float

class ResearchReport(BaseModel):
    answer: str = Field(description="The comprehensive, synthesized answer formatted in Markdown")
    sources: List[int] = Field(description="List of source numbers used", default=[])
    reasoning_steps: Optional[List[ReasoningStep]] = Field(description="Multi-step reasoning process", default=None)
    confidence_score: float = Field(description="Overall confidence in the answer (0-1)", ge=0.0, le=1.0)
    query_complexity: str = Field(description="Simple, Moderate, or Complex")

class QueryResponse(BaseModel):
    report: ResearchReport
    source_documents: List[Dict[str, Any]]
    processing_time: float
    cache_hit: bool

# --- Enhanced Helper Functions ---
def demo_optimized_text_splitter(docs: List[Document]) -> List[Document]:
    """Optimized text splitting for demo performance"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800 if DEMO_MODE else 1000,
            chunk_overlap=150 if DEMO_MODE else 200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        splits = text_splitter.split_documents(docs)
        
        for i, split in enumerate(splits):
            split.metadata.update({
                "chunk_id": i,
                "chunk_size": len(split.page_content),
                "processing_timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"Text splitting completed: {len(splits)} chunks created")
        return splits
    except Exception as e:
        logger.error(f"Error in text splitting: {e}")
        raise e

def analyze_query_complexity(question: str) -> str:
    """Analyze query complexity to determine processing approach"""
    try:
        question_lower = question.lower()
        
        complex_indicators = [
            "compare", "contrast", "analyze", "evaluate", "relationship between",
            "how does", "what causes", "explain why", "multi", "several", "various"
        ]
        
        moderate_indicators = [
            "what are", "list", "describe", "summarize", "overview", "examples"
        ]
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in question_lower)
        moderate_count = sum(1 for indicator in moderate_indicators if indicator in question_lower)
        
        if complex_count >= 2 or len(question.split()) > 15:
            return "Complex"
        elif moderate_count >= 1 or len(question.split()) > 8:
            return "Moderate"
        else:
            return "Simple"
    except Exception as e:
        logger.error(f"Error in query complexity analysis: {e}")
        return "Simple"  # Default fallback

def format_docs_with_enhanced_metadata(docs: List[Document]) -> str:
    """Enhanced document formatting with better context"""
    try:
        formatted_strings = []
        for i, doc in enumerate(docs):
            source_info = doc.metadata.get("source", "Unknown").split('/')[-1]
            page_info = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", i)
            
            content = doc.page_content.replace('\n', ' ').strip()
            
            formatted_strings.append(
                f"Source [{i+1}] (Chunk {chunk_id}):\n"
                f"Document: {source_info}\n"
                f"Page: {page_info}\n"
                f"Content: {content}\n"
                f"Relevance Context: This chunk contains information about the document's content."
            )
        
        return "\n\n".join(formatted_strings)
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        return "Error formatting document context"

async def multi_step_reasoning(question: str, vectorstore, max_steps: int = 2) -> Dict[str, Any]:
    """Implement multi-step reasoning for complex queries - optimized for demo"""
    try:
        reasoning_steps = []
        accumulated_context = []
        retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
        
        # Initialize LLM with error handling
        if not os.getenv("CLOUDRIFT_API_KEY"):
            raise ValueError("CLOUDRIFT_API_KEY not found in environment variables")
            
        llm = ChatOpenAI(
            model="Qwen/Qwen3-Next-80B-A3B-Thinking",
            api_key=os.getenv("CLOUDRIFT_API_KEY"),
            base_url="https://inference.cloudrift.ai/v1",
            temperature=0.1
        )
        
        current_question = question
        
        for step in range(max_steps):
            try:
                if step > 0:
                    sub_question_prompt = f"""
                    Based on the original question: "{question}"
                    And the previous findings: {json.dumps([step.partial_answer for step in reasoning_steps], indent=2)}
                    
                    What specific sub-question should be asked next to get closer to a complete answer?
                    Respond with just the sub-question, nothing else.
                    """
                    
                    sub_question_response = await llm.ainvoke(sub_question_prompt)
                    current_question = sub_question_response.content.strip()
                else:
                    current_question = question
                
                # Retrieve documents for current question
                docs = await asyncio.get_event_loop().run_in_executor(
                    executor, retriever.invoke, current_question
                )
                
                # Generate partial answer
                context = format_docs_with_enhanced_metadata(docs)
                
                partial_answer_prompt = f"""
                Question: {current_question}
                Context: {context}
                
                Provide a focused answer to this specific question based only on the given context.
                If the context doesn't contain enough information, state that clearly.
                Keep your response concise but informative.
                """
                
                partial_response = await llm.ainvoke(partial_answer_prompt)
                
                # Calculate confidence score
                confidence = min(1.0, len([doc for doc in docs if len(doc.page_content) > 100]) / 4.0)
                
                reasoning_steps.append(ReasoningStep(
                    step_number=step + 1,
                    sub_question=current_question,
                    retrieved_sources=list(range(len(accumulated_context), len(accumulated_context) + len(docs))),
                    partial_answer=partial_response.content,
                    confidence_score=confidence
                ))
                
                accumulated_context.extend(docs)
                
                # Check if we have enough information
                if confidence > 0.7 and len(partial_response.content) > 150:
                    break
                    
            except Exception as step_error:
                logger.error(f"Error in reasoning step {step + 1}: {step_error}")
                break
        
        return {
            "reasoning_steps": reasoning_steps,
            "all_retrieved_docs": accumulated_context,
            "final_confidence": sum(step.confidence_score for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.0
        }
    except Exception as e:
        logger.error(f"Error in multi-step reasoning: {e}")
        raise e

# --- Enhanced API Endpoints ---
@app.post("/upload", response_model=UploadResponse, summary="Upload and process a PDF document")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    start_time = datetime.now()
    tmp_file_path = None
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")

        # Load and process document
        loader = PyPDFLoader(tmp_file_path)
        docs = await asyncio.get_event_loop().run_in_executor(executor, loader.load)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the PDF")
        
        total_pages = len(set(doc.metadata.get("page", 0) for doc in docs))
        
        # Demo-optimized text splitting
        splits = await asyncio.get_event_loop().run_in_executor(executor, demo_optimized_text_splitter, docs)
        
        if not splits:
            raise HTTPException(status_code=400, detail="Document could not be split into chunks")
        
        # Use optimized embeddings
        embeddings = get_optimized_embeddings()
        
        vectorstore = await asyncio.get_event_loop().run_in_executor(
            executor, FAISS.from_documents, splits, embeddings
        )

        # Generate unique ID and store
        doc_id = str(uuid.uuid4())
        processing_time = (datetime.now() - start_time).total_seconds()
        
        metadata = DocumentMetadata(
            filename=file.filename,
            file_size=file_size,
            total_pages=total_pages,
            total_chunks=len(splits),
            processing_time=processing_time,
            upload_timestamp=start_time
        )
        
        document_stores[doc_id] = DocumentStore(
            vectorstore=vectorstore,
            metadata=metadata.dict(),
            created_at=start_time
        )

        logger.info(f"Successfully processed document {file.filename} with {len(splits)} chunks in {processing_time:.2f}s")

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            message=f"Document processed successfully! Created {len(splits)} chunks from {total_pages} pages in {processing_time:.2f} seconds.",
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/query", response_model=QueryResponse, summary="Query document with enhanced error handling")
async def query_agent(request: QueryRequest):
    logger.info(f"Received query request: doc_id={request.doc_id}, question='{request.question[:50]}...'")
    
    if request.doc_id not in document_stores:
        logger.error(f"Document ID not found: {request.doc_id}")
        raise HTTPException(status_code=404, detail="Document ID not found. Please upload the document first.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = datetime.now()
    
    # Check cache first
    cache_key = f"{request.doc_id}:{hash(request.question)}"
    if cache_key in query_cache:
        cached_result = query_cache[cache_key].copy()
        cached_result["cache_hit"] = True
        logger.info(f"Cache hit for query: {request.question[:30]}")
        return QueryResponse(**cached_result)

    try:
        doc_store = document_stores[request.doc_id]
        doc_store.access_count += 1
        doc_store.last_accessed = datetime.now()
        
        vectorstore = doc_store.vectorstore
        query_complexity = analyze_query_complexity(request.question)
        
        logger.info(f"Query complexity determined: {query_complexity}")
        
        # Validate API key
        if not os.getenv("CLOUDRIFT_API_KEY"):
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="Qwen/Qwen3-Next-80B-A3B-Thinking",
            api_key=os.getenv("CLOUDRIFT_API_KEY"),
            base_url="https://inference.cloudrift.ai/v1",
            temperature=0.1
        )

        # Process query based on complexity
        if query_complexity == "Complex" and request.enable_multi_step:
            logger.info("Using multi-step reasoning")
            reasoning_result = await multi_step_reasoning(
                request.question, 
                vectorstore, 
                request.max_reasoning_steps
            )
            
            reasoning_steps = reasoning_result["reasoning_steps"]
            all_docs = reasoning_result["all_retrieved_docs"]
            confidence = reasoning_result["final_confidence"]
            
            # Synthesize final answer
            synthesis_prompt = f"""
            Original Question: {request.question}
            
            Reasoning Steps:
            {json.dumps([step.dict() for step in reasoning_steps], indent=2)}
            
            Synthesize a comprehensive, coherent answer that incorporates insights from all reasoning steps.
            Format your response as a detailed research report in Markdown.
            Keep it concise but thorough.
            """
            
            final_response = await llm.ainvoke(synthesis_prompt)
            
            # Extract source numbers used
            used_sources = []
            for step in reasoning_steps:
                used_sources.extend(step.retrieved_sources)
            used_sources = list(set(used_sources))
            
            final_answer = final_response.content
            
        else:
            logger.info("Using single-step retrieval")
            # Use traditional single-step retrieval
            retriever = vectorstore.as_retriever(search_kwargs={'k': DEFAULT_RETRIEVAL_K})
            docs = await asyncio.get_event_loop().run_in_executor(
                executor, retriever.invoke, request.question
            )
            
            if not docs:
                raise HTTPException(status_code=404, detail="No relevant documents found for the query")
            
            # Create a simpler prompt that doesn't require JSON parsing
            context = format_docs_with_enhanced_metadata(docs)
            
            simple_prompt = f"""You are an expert research analyst. Based ONLY on the provided context, provide a comprehensive answer to the question.

Context:
{context}

Question: {request.question}

Please provide a detailed answer using only the information from the context above. If the context doesn't contain enough information, clearly state that limitation. Format your response in Markdown for better readability."""
            
            try:
                response = await llm.ainvoke(simple_prompt)
                final_answer = response.content
                
                all_docs = docs
                reasoning_steps = None
                confidence = min(1.0, len([doc for doc in docs if len(doc.page_content) > 100]) / DEFAULT_RETRIEVAL_K)
                used_sources = list(range(1, len(docs) + 1))
                
            except Exception as llm_error:
                logger.error(f"LLM invocation error: {llm_error}")
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(llm_error)}")

        # Prepare source documents
        source_data = [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source", "").split('/')[-1]
            }
            for doc in all_docs
        ]

        # Create final report
        report = ResearchReport(
            answer=final_answer,
            sources=used_sources,
            reasoning_steps=reasoning_steps,
            confidence_score=confidence,
            query_complexity=query_complexity
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "report": report,
            "source_documents": source_data,
            "processing_time": processing_time,
            "cache_hit": False
        }

        # Cache the result
        if len(query_cache) < MAX_CACHE_SIZE:
            query_cache[cache_key] = response.copy()
        
        logger.info(f"Successfully processed query for doc {request.doc_id} in {processing_time:.2f}s")

        return QueryResponse(**response)

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error during query processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error during query processing: {str(e)}")

@app.get("/document/{doc_id}/info", summary="Get document information")
async def get_document_info(doc_id: str):
    if doc_id not in document_stores:
        raise HTTPException(status_code=404, detail="Document ID not found.")
    
    doc_store = document_stores[doc_id]
    return {
        "doc_id": doc_id,
        "metadata": doc_store.metadata,
        "access_count": doc_store.access_count,
        "last_accessed": doc_store.last_accessed,
        "created_at": doc_store.created_at
    }

@app.delete("/document/{doc_id}", summary="Delete a document")
async def delete_document(doc_id: str):
    if doc_id not in document_stores:
        raise HTTPException(status_code=404, detail="Document ID not found.")
    
    del document_stores[doc_id]
    keys_to_remove = [key for key in query_cache.keys() if key.startswith(f"{doc_id}:")]
    for key in keys_to_remove:
        del query_cache[key]
    
    return {"message": "Document deleted successfully"}

@app.get("/stats", summary="Get system statistics")
async def get_stats():
    return {
        "total_documents": len(document_stores),
        "cache_entries": len(query_cache),
        "total_access_count": sum(store.access_count for store in document_stores.values()),
        "demo_mode": DEMO_MODE,
        "memory_usage": {
            "documents": len(document_stores),
            "cache_size": len(query_cache)
        }
    }

# Health check endpoint
@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Railway's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
