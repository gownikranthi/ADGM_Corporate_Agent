ADGM Corporate Agent
An intelligent AI-powered legal assistant designed to review and validate legal documents for compliance within the Abu Dhabi Global Market (ADGM) jurisdiction. This system leverages a multi-agent architecture and Retrieval-Augmented Generation (RAG) to ensure accuracy and efficiency.

Overview and Agentic Flow ⚙️
The system operates as an agentic pipeline, where a central Orchestrator manages the flow of information between specialized, task-specific agents. The process is designed to mimic a human workflow, with each agent acting as an expert in its field.

The agentic flow follows these steps:

Document Ingestion: A user uploads one or more .docx files through the Gradio UI.

Orchestration: The run_corporate_agent function acts as the Orchestrator. It receives the documents and sequentially delegates tasks to the specialized agents.

Analysis & Validation: The agents perform their tasks, from parsing the documents to detecting red flags and verifying checklists.

Output Generation: A final agent gathers all the reports and generates a marked-up .docx file and a structured JSON summary.

User Output: The final outputs are displayed in the UI for the user to download and review.

The Agents and Their Roles 🤖
The system is composed of several key agents, each with a specific responsibility:

Document Parsing Agent: This agent's sole purpose is to ingest .docx files and extract their raw text content. It uses the python-docx library to handle the document format and passes the extracted text to the next agents in the pipeline.

Document Classification Agent: Using the gpt-4o LLM, this agent analyzes the parsed text to determine the document's type (e.g., Articles of Association, Employment Contract). This classification is crucial for the Checklist Verification Agent.

Checklist Verification Agent: This agent compares the classified documents against a predefined list of mandatory documents (stored in data/checklists.json) for a specific legal process. It also uses the LLM to intelligently infer the legal process based on the documents provided. It then reports any missing documents in the final output.

RAG Validation Agent: This is the core of the system's intelligence. It uses Retrieval-Augmented Generation (RAG) to perform a legal review. It searches a knowledge base of official ADGM regulations (stored in a FAISS vector store) and uses the retrieved information to instruct the gpt-4o LLM to detect red flags, inconsistencies, and provide legally compliant suggestions.

Document Generation Agent: The final agent in the pipeline. It takes the original .docx file and the detailed report from the RAG Validation Agent to produce the final deliverables. It highlights flagged sections, inserts contextual comments with citations, and saves the file for the user to download.

Usability and Technologies 💻
Usability: The agent is designed for legal professionals, business owners, and administrators. The Gradio UI is simple and intuitive, requiring no special training to use.

Core Technologies: The system is built on a modern Python stack using:

LLM: OpenAI's gpt-4o for complex reasoning and content generation.

RAG: LangChain is used to orchestrate the RAG pipeline, with FAISS serving as the efficient vector store.

Agentic Framework: The system follows an agentic design pattern, with specialized agents communicating to complete a complex task.

UI: Gradio provides a fast and easy-to-use web interface for demonstration.

Submission Checklist
GitHub Repository: This codebase.

README: This document.

Example Document: A before and after version of a reviewed .docx file.

Structured Output: The generated JSON report.

Demo: A screenshot or video demonstrating the UI in action.

Setup Instructions
1. Prerequisites
Python 3.10+

An OpenAI API key

2. Installation
Clone the repository and navigate to the project directory.

Set up a virtual environment and install dependencies:

Bash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Create a .env file in the root directory and add your API key:

OPENAI_API_KEY="sk-your-openai-api-key-here"
3. Usage
Build the RAG Knowledge Base:
Run the following command once to download documents and create the vector store:

Bash

python app/rag_pipeline.py
Run the Application:
Launch the Gradio web interface from the project's root directory:

Bash

python ui/app.py
Your application will be accessible at the local URL displayed in the terminal.# ADGM_Corparate_Agent
