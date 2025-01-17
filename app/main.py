from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from typing import Dict
from pdfminer.high_level import extract_text
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.file import FileTools
import asyncio

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

# Function to create exam analysis workflow
def create_exam_analysis_workflow(api_key: str, syllabus_text: str, question_papers: Dict[int, str]):
    os.environ["GROQ_API_KEY"] = api_key
    groq_model = Groq(id="llama-3.1-70b-versatile")

    # Define agents
    syllabus_agent = Agent(
        name="syllabus parser",
        role="parse the given string to determine the subject and syllabus.",
        model=groq_model,
        instructions=["Always respond in a well-structured format including all 5 units and their detailed topics."]
    )
    
    questionpaper_agent = Agent(
        name="question paper parser",
        role="parse the given dictionary to detect questions from each section",
        model=groq_model,
        instructions=[
            "Return a proper dictionary consisting of all the questions mentioned in the given question paper dictionary with year as key and questions string as value.",
            "Return the questions only, exclude any other details."
        ],
    )
    
    exam_analyser_agent = Agent(
        name="ai exam analyser",
        role="Analyze syllabus and question papers.",
        model=groq_model,
        instructions=[
            "Use syllabus of subject and analyse question papers to determine the most important and frequently asked topics from each unit.",
            "Always respond in a structured format- Unit Name: important topics.",
            "Always include the number of times the topic has appeared in the question papers."
        ]
    )
    
    question_generator_agent = Agent(
        name="question generator",
        role="Generate probable exam questions.",
        model=groq_model,
        instructions=[
            "Generate 10-15 probable questions for upcoming exam based on analysis of past papers and important topics",
            "Include both short answer (2 marks) and long answer (10 marks) questions",
            "Focus on frequently tested topics and important concepts",
            "Format the output as a proper model question paper in markdown",
            "Ensure questions are unique and not directly copied from past papers",
            "Refer to the type and quality of questions from previous year question papers provided",
            "Mark the weightage (2/10 marks) clearly for each question"
        ],
    )

    async def run_workflow():
        try:
            syllabus_result = syllabus_agent.run(f"Parse and structure this syllabus content: {syllabus_text}")
            questions_result = questionpaper_agent.run(f"Parse and structure these question papers: {str(question_papers)}")
            analysis_result = exam_analyser_agent.run(
                f"""Analyse the following:
                Syllabus structure: {syllabus_result.content}\n
                Question paper history: {questions_result.content}
                
                Provide analysis of important topics and their frequency."""
            )
            practice_questions = question_generator_agent.run(
                f"""Based on the following information:
                Question paper patterns: {questions_result.content}
                Topic frequency analysis: {analysis_result.content}
                
                Generate a model question paper with 10-15 questions following these guidelines:
                1. Include both short (5 marks) and long answer (10 marks) questions
                2. Focus on topics that appear frequently in past papers
                3. Cover important topics from all units
                4. Include theoretical and practical questions
                5. Format as a proper question paper with sections and marks
                """
            )
            return {
                "exam_analysis": analysis_result.content,
                "practice_questions": practice_questions.content,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during analysis: {e}")

    return run_workflow

# Endpoint to handle file uploads and analysis
@app.post("/analyze/")
async def analyze(api_key: str = Form(...), syllabus_file: UploadFile = File(...), question_files: list[UploadFile] = File(...)):
    print("Received API Key:", api_key)
    print("Number of question files:", len(question_files))
    
    # Extract text from uploaded files
    syllabus_text = extract_text_from_pdf(syllabus_file.file)
    question_papers = {idx: extract_text_from_pdf(file.file) for idx, file in enumerate(question_files)}

    if not syllabus_text or not any(question_papers.values()):
        raise HTTPException(status_code=400, detail="Failed to extract text from uploaded files.")

    # Create and run workflow
    workflow = create_exam_analysis_workflow(api_key, syllabus_text, question_papers)
    results = await workflow()

    # Log the results for debugging
    print("Results:", results)

    return results

# Serve a simple HTML form for testing (optional)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Xamify API</title>
        </head>
        <body>
            <h1>Welcome to the Xamify API!</h1>
            <p>Use the /analyze endpoint to upload your syllabus and question papers.</p>
        </body>
    </html>
    """ 