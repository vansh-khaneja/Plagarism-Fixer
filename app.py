from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from plagchecker import PlagiarismChecker
import uvicorn
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Plagiarism Finder & Remover API", version="1.0.0")

# Global checker instance
checker = None

# Serve the frontend HTML file
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML file"""
    return FileResponse("index.html")

# Add CORS middleware for frontend requests
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PlagiarismRequest(BaseModel):
    content: str

class RewriteRequest(BaseModel):
    chunk_text: str
    num_suggestions: int = 3

class RewriteResponse(BaseModel):
    original_text: str
    suggestions: List[str]
    processing_time: float

class ChunkResult(BaseModel):
    chunk_id: int
    word_count: int
    content: str
    is_plagiarized: bool
    similarity_score: float
    source_title: Optional[str] = None
    source_url: Optional[str] = None

class PlagiarismResponse(BaseModel):
    total_words: int
    total_chunks: int
    chunks_processed: int
    chunks_with_plagiarism: int
    overall_plagiarism_score: float
    chunk_details: List[ChunkResult]
    processing_time: float

@app.post("/check-plagiarism", response_model=PlagiarismResponse)
async def check_plagiarism(request: PlagiarismRequest):
    """
    Check content for plagiarism and return detailed results
    """
    import time
    start_time = time.time()
    
    try:
        # Get API credentials from environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            raise HTTPException(
                status_code=400, 
                detail="API credentials not found. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
            )
        
        # Get configuration from environment variables
        chunk_size = int(os.getenv("CHUNK_SIZE", "80"))
        overlap_percentage = float(os.getenv("OVERLAP_PERCENTAGE", "0.15"))
        similarity_threshold = int(os.getenv("SIMILARITY_THRESHOLD", "50"))
        max_chunks = int(os.getenv("MAX_CHUNKS", "0")) if os.getenv("MAX_CHUNKS", "0") != "0" else None
        delay_between_requests = float(os.getenv("DELAY_BETWEEN_REQUESTS", "1.0"))
        
        # Initialize checker with environment credentials and config
        global checker
        checker = PlagiarismChecker(
            api_key=api_key,
            search_engine_id=search_engine_id,
            chunk_size=chunk_size,
            overlap_percentage=overlap_percentage
        )
        
        # Check for plagiarism
        result = checker.check_plagiarism(
            article_text=request.content,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks,
            delay_between_requests=delay_between_requests
        )
        
        # Process chunk details
        chunk_details = []
        chunks = checker.create_chunks(request.content)
        
        # Create detailed chunk results
        for i, chunk in enumerate(chunks):
            # Check if this chunk has plagiarism
            plagiarized_chunk = None
            for match in result['matches']:
                if match['chunk_index'] == i:
                    plagiarized_chunk = match
                    break
            
            chunk_result = ChunkResult(
                chunk_id=i,
                word_count=chunk['word_count'],
                content=chunk['text'],
                is_plagiarized=plagiarized_chunk is not None,
                similarity_score=plagiarized_chunk['similarity'] if plagiarized_chunk else 0.0,
                source_title=plagiarized_chunk['source_title'] if plagiarized_chunk else None,
                source_url=plagiarized_chunk['source_url'] if plagiarized_chunk else None
            )
            chunk_details.append(chunk_result)
        
        processing_time = time.time() - start_time
        
        return PlagiarismResponse(
            total_words=len(request.content.split()),
            total_chunks=result['total_chunks'],
            chunks_processed=result['checked_chunks'],
            chunks_with_plagiarism=result['plagiarized_chunks'],
            overall_plagiarism_score=result['plagiarism_percentage'],
            chunk_details=chunk_details,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing plagiarism check: {str(e)}")

@app.post("/rewrite-chunk", response_model=RewriteResponse)
async def rewrite_chunk(request: RewriteRequest):
    """
    Rewrite a plagiarized chunk to avoid plagiarism using OpenAI
    """
    import time
    start_time = time.time()
    
    try:
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        
        # Initialize OpenAI client with new API
        client = OpenAI(api_key=openai_api_key)
        
        # Create prompt for rewriting
        prompt = f"""
        Rewrite the following text to avoid plagiarism while maintaining the same meaning and quality. 
        Provide {request.num_suggestions} different versions that are:
        1. Substantially different in wording and structure
        2. Maintain the same core meaning and information
        3. Sound natural and well-written
        4. Avoid any direct copying of phrases or sentences
        
        Original text: "{request.chunk_text}"
        
        Please provide {request.num_suggestions} rewritten versions, each on a new line starting with "Version X:"
        """
        
        # Call OpenAI API using new format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional writer who specializes in rewriting content to avoid plagiarism while maintaining quality and meaning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Parse the response
        ai_response = response.choices[0].message.content.strip()
        
        # Extract suggestions from the response
        suggestions = []
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Version ') and ':' in line:
                # Extract text after the colon
                suggestion = line.split(':', 1)[1].strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        # If we didn't get enough suggestions, try to split by other patterns
        if len(suggestions) < request.num_suggestions:
            # Try splitting by numbered lists
            parts = ai_response.split('\n\n')
            for part in parts:
                part = part.strip()
                if part and not part.startswith('Version '):
                    suggestions.append(part)
        
        # Ensure we have the requested number of suggestions
        while len(suggestions) < request.num_suggestions and len(suggestions) > 0:
            suggestions.append(suggestions[-1])  # Duplicate last suggestion if needed
        
        # Limit to requested number
        suggestions = suggestions[:request.num_suggestions]
        
        processing_time = time.time() - start_time
        
        return RewriteResponse(
            original_text=request.chunk_text,
            suggestions=suggestions,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rewriting chunk: {str(e)}")

@app.get("/api-info")
async def api_info():
    """
    API information endpoint
    """
    return {
        "message": "Plagiarism Finder & Remover API",
        "version": "1.0.0",
        "frontend": "Available at http://5.223.43.159:8000/",
        "endpoints": {
            "GET /": "Frontend interface",
            "POST /check-plagiarism": "Check content for plagiarism",
            "POST /rewrite-chunk": "Rewrite plagiarized chunk to avoid plagiarism",
            "GET /api-info": "API information",
            "GET /health": "Health check"
        },
        "environment_variables": {
            "GOOGLE_API_KEY": "✅ Set" if os.getenv("GOOGLE_API_KEY") else "❌ Not set",
            "GOOGLE_SEARCH_ENGINE_ID": "✅ Set" if os.getenv("GOOGLE_SEARCH_ENGINE_ID") else "❌ Not set",
            "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set",
            "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "80"),
            "OVERLAP_PERCENTAGE": os.getenv("OVERLAP_PERCENTAGE", "0.15"),
            "SIMILARITY_THRESHOLD": os.getenv("SIMILARITY_THRESHOLD", "50"),
            "MAX_CHUNKS": os.getenv("MAX_CHUNKS", "0"),
            "DELAY_BETWEEN_REQUESTS": os.getenv("DELAY_BETWEEN_REQUESTS", "1.0")
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    api_key_status = "✅ Set" if os.getenv("GOOGLE_API_KEY") else "❌ Not set"
    search_engine_status = "✅ Set" if os.getenv("GOOGLE_SEARCH_ENGINE_ID") else "❌ Not set"
    openai_status = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set"
    
    return {
        "status": "healthy", 
        "message": "API is running",
        "credentials": {
            "google_api_key": api_key_status,
            "google_search_engine_id": search_engine_status,
            "openai_api_key": openai_status
        },
        "configuration": {
            "chunk_size": os.getenv("CHUNK_SIZE", "80"),
            "overlap_percentage": os.getenv("OVERLAP_PERCENTAGE", "0.15"),
            "similarity_threshold": os.getenv("SIMILARITY_THRESHOLD", "50"),
            "max_chunks": os.getenv("MAX_CHUNKS", "0"),
            "delay_between_requests": os.getenv("DELAY_BETWEEN_REQUESTS", "1.0")
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
