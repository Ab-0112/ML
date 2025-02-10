import os
import logging
import time
import subprocess
import eventlet
from flask import Flask
from flask_socketio import SocketIO, emit
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", ping_timeout=300, ping_interval=30)

logging.basicConfig(level=logging.INFO)

# Global variables
vector_db = None
task_chain = None
DESCRIPTIONS_FILE = 'description.txt'
DB_DIRECTORY = "chroma_db"

# =========================== UTILITY FUNCTIONS =========================== #
def extract_descriptions(file_path):
    """Extracts Job and Company Descriptions from description.txt file"""
    if not os.path.exists(file_path):
        return {"error": "🚨 Error: File not found!"}

    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read().strip()

    job_marker = "# Job Description"
    company_marker = "# Company Description"

    job_start_index = content.find(job_marker)
    company_start_index = content.find(company_marker)

    if job_start_index == -1 or company_start_index == -1:
        return {"error": "🚨 Error: Missing job or company description markers."}

    return {
        "job_description": content[job_start_index + len(job_marker):company_start_index].strip(),
        "company_description": content[company_start_index + len(company_marker):].strip()
    }

def ensure_model_available(model_name):
    """Ensures Ollama model is available before running queries"""
    try:
        logging.info(f"🔍 Checking model availability: {model_name}")
        subprocess.run(["ollama", "pull", model_name], check=True)
        logging.info(f"✅ Model '{model_name}' is ready.")
    except subprocess.CalledProcessError:
        logging.error(f"🚨 Failed to pull model '{model_name}'. Ensure Ollama is running.")
        raise RuntimeError(f"Model '{model_name}' not available. Please start Ollama.")

# =========================== TASK CHAIN INITIALIZATION =========================== #
def initialize_task_chain():
    """Initializes the RAG-based pipeline for task generation"""
    global vector_db, task_chain

    descriptions = extract_descriptions(DESCRIPTIONS_FILE)
    if "error" in descriptions:
        logging.error(descriptions["error"])
        return False

    job_description = descriptions["job_description"]

    ensure_model_available("nomic-embed-text")

    # Load Job Description into ChromaDB
    documents = [Document(page_content=job_description)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="tasks-collection",
        persist_directory=DB_DIRECTORY  # Ensures data is stored persistently
    )

    task_prompt = ChatPromptTemplate.from_template(
        """
        Design realistic, scenario-based interview tasks that assess a candidate’s ability to perform job-specific tasks. Each task must simulate a real-world problem they are likely to face in their role.

        Input:

        Job Description: {job_description}
        Company Description: {company_description}
        Output:
        Generate 6 interview tasks, ensuring each task follows this structure:

        Task Title : A concise, descriptive title.
        Scenario/Context : A practical, real-world situation relevant to the job role.
        Objective : The key skill or competency being tested (e.g., problem-solving, coding, communication, decision-making).
        Task Description : Clear, actionable instructions without reliance on guides or tutorials.
        Resources : Only tools, platforms, or datasets the candidate would actually use in the job (e.g., CRM, SQL database, Jira, Python scripts).
        Estimated Time : The time needed to complete the task (≤ 90 minutes).
        Evaluation Criteria : How performance will be judged (e.g., accuracy, efficiency, creativity, clarity).
        
        Additional Notes:

        Keep all tasks realistic and job-relevant.
        Ensure candidates can complete tasks using workplace tools—avoid theoretical exercises.
        Balance technical, problem-solving, and soft skills assessments.
        Avoid tasks that require learning new tools outside of what’s commonly used in the role.
        """
    )

    task_chain = task_prompt | ChatOllama(model="mistral", streaming=True) | StrOutputParser()
    return True

# =========================== SOCKET.IO EVENTS =========================== #
@socketio.on('connect')
def handle_connect():
    """Handles WebSocket client connection."""
    logging.info("✅ Client connected via WebSocket.")
    emit('message', {'data': '🔌 Connected to WebSocket server!'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles WebSocket client disconnection."""
    logging.info("❌ Client disconnected.")

import re

@socketio.on('generate_task')
def handle_generate_task():
    """Generates structured tasks and streams them sentence-by-sentence to the frontend in real-time"""
    try:
        if task_chain is None:
            emit('message', {'data': '❌ Task chain is not initialized!'})
            return

        descriptions = extract_descriptions(DESCRIPTIONS_FILE)
        if "error" in descriptions:
            emit('message', {'data': descriptions["error"]})
            return

        job_description = descriptions["job_description"]
        company_description = descriptions.get("company_description", "")

        logging.info(f"⚡ Generating tasks with job description: {job_description[:50]}...")
        emit('message', {'data': '🔄 Generating tasks...'}, broadcast=True)

        start_time = time.time()

        # ✅ Streaming Response Instead of Waiting
        response_generator = task_chain.stream({
            "job_description": job_description,
            "company_description": company_description
        })

        sentence_buffer = ""

        for chunk in response_generator:
            sentence_buffer += chunk  # ✅ Accumulate text

            # ✅ Split into sentences using regex
            sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)

            for sentence in sentences[:-1]:  # ✅ Process complete sentences only
                sentence = sentence.strip()
                if sentence:
                    logging.info(f"📩 Sending sentence: {sentence}")  # ✅ Logs sentence
                    emit('message', {'data': sentence}, broadcast=True)  # ✅ Emit sentence
                    time.sleep(0.1)  # ✅ Faster real-time updates

            sentence_buffer = sentences[-1]  # ✅ Keep incomplete sentence for next chunk

        # ✅ Emit last remaining sentence
        if sentence_buffer.strip():
            logging.info(f"📩 Sending final sentence: {sentence_buffer.strip()}")
            emit('message', {'data': sentence_buffer.strip()}, broadcast=True)

        elapsed_time = time.time() - start_time

        logging.info(f"✅ Task generation completed in {elapsed_time:.2f} seconds")
        emit('task_completion', {'elapsed_time': elapsed_time}, broadcast=True)

    except Exception as e:
        logging.error(f"🚨 Task generation failed: {e}")
        emit('message', {'data': f"Error: {str(e)}"}, broadcast=True)

# =========================== SERVER START (WITH DEBUG MODE) =========================== #
if __name__ == '__main__':
    logging.info("⚡ Starting WebSocket server...")

    # Initialize task chain at startup
    if initialize_task_chain():
        logging.info("✅ Task chain initialized at startup!")
    else:
        logging.error("🚨 Task chain initialization failed!")

    # ✅ Run with debug mode enabled
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
