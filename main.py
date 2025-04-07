# main.py
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
from googleapiclient.discovery import build
from dotenv import load_dotenv
load_dotenv()  
app = FastAPI(title="AI Profile & Email Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Environment variables (in production, set these as actual env vars)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-google-api-key")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "your-google-cse-id")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key")

# Models
class ChatRequest(BaseModel):
    message: str
    task_type: str  # "profile_summary" or "custom_email"
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    search_results: Optional[List[Dict[str, Any]]] = None

# Google Search Function
def google_search(search_term, api_key=GOOGLE_API_KEY, cse_id=GOOGLE_CSE_ID, **kwargs):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res.get('items', [])
    except Exception as e:
        print(f"Google Search Error: {e}")
        return []

# LLM Processing Function
def process_with_llm(search_results, task_type, context=None):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Format the search results for the prompt
    formatted_results = "\n\n".join([
        f"Title: {item.get('title', 'No title')}\n"
        f"Link: {item.get('link', 'No link')}\n"
        f"Snippet: {item.get('snippet', 'No snippet')}"
        for item in search_results[:5]  # Limit to top 5 results
    ])
    
    # Add context-specific information
    context_info = ""
    if context:
        if task_type == "profile_summary" and context.get("person_info"):
            context_info = f"Additional information about the person:\n{context.get('person_info')}\n\n"
        elif task_type == "custom_email" and context.get("email_purpose"):
            context_info = f"Purpose of the email: {context.get('email_purpose')}\n\n"
    
    # Create prompt based on task type
    if task_type == "profile_summary":
        prompt = (
            f"Based on the following information, create a professional profile summary "
            f"that highlights key achievements, skills, and experiences in a coherent narrative.\n\n"
            f"{context_info}Search Results:\n{formatted_results}"
        )
    elif task_type == "custom_email":
        prompt = (
            f"Based on the following information, draft a personalized email "
            f"that is professional, engaging, and tailored to the recipient.\n\n"
            f"{context_info}Search Results:\n{formatted_results}"
        )
    else:
        return "Unsupported task type"
    
    try:
        # Get response from LLM
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    except Exception as e:
        print(f"LLM Processing Error: {e}")
        return f"Error processing with LLM: {str(e)}"

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Extract search terms from the message
    search_term = request.message
    
    # Perform Google search
    search_results = google_search(search_term)
    
    if not search_results:
        return ChatResponse(
            response="I couldn't find relevant information. Please try a different search term.",
            search_results=[]
        )
    
    # Process with LLM
    llm_response = process_with_llm(search_results, request.task_type, request.context)
    
    return ChatResponse(
        response=llm_response,
        search_results=[{
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "")
        } for item in search_results[:3]]  # Return top 3 results for display
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)