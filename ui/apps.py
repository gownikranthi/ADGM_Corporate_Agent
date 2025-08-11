import sys
import os
import gradio as gr
import json

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import run_corporate_agent

def process_documents(files):
    if not files:
        return "Please upload at least one document.", "{}", None
    
    file_paths = [file.name for file in files]
    
    try:
        reviewed_doc_paths, json_report = run_corporate_agent(file_paths)
        
        if reviewed_doc_paths and json_report:
            return "Processing complete!", json.dumps(json_report, indent=2), reviewed_doc_paths
        else:
            return "An error occurred during processing.", "{}", None
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", "{}", None

css_styles = """
h1 {
    text-align: center;
    color: #1a202c;
    font-family: 'Arial', sans-serif;
    font-size: 2.5rem;
}
.gr-button {
    background-color: #2b6cb0;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1rem;
    font-weight: 600;
}
.gr-file {
    border-radius: 8px;
    border: 2px dashed #a0aec0;
    background-color: #f7fafc;
    padding: 20px;
}
.gr-json {
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
    font-family: 'Courier New', monospace;
}
.gr-status {
    font-size: 1.1rem;
    font-weight: 500;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css_styles, title="ADGM Corporate Agent") as demo:
    gr.Markdown("# ADGM Corporate Agent", elem_id="title-heading")
    gr.Markdown("An AI-powered legal assistant to review and validate documents for ADGM compliance.")
    
    with gr.Tabs():
        with gr.TabItem("Upload & Analyze"):
            with gr.Row():
                file_input = gr.File(
                    label="Upload .docx and .pdf Documents",
                    file_count="multiple",
                    file_types=[".docx", ".pdf"],
                    interactive=True,
                    height=200,
                    elem_id="file-input"
                )
            
            with gr.Row():
                submit_button = gr.Button("Analyze Documents", variant="primary")

            gr.Markdown("---")
            
            status_output = gr.Textbox(label="Status", interactive=False, elem_id="status-output")
            
            with gr.Column():
                gr.Markdown("### 📄 Reviewed Documents")
                file_output = gr.File(label="Download Reviewed Documents", file_count="multiple", interactive=False, elem_id="file-output")
                
                gr.Markdown("### 📊 Analysis Report")
                json_output = gr.JSON(label="Structured Report (JSON)", elem_id="json-output")
        
    submit_button.click(
        fn=process_documents,
        inputs=file_input,
        outputs=[status_output, json_output, file_output]
    )
    
if __name__ == "__main__":
    demo.launch()