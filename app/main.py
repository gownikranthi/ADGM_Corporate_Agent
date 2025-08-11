# File: Agentic_ADGM/app/main.py

import os
import json
# CORRECTED: Import the new function name 'parse_document' instead of 'parse_docx'
from .agents import parse_document, classify_document, rag_validate_document, verify_checklist, generate_output_docs

def run_corporate_agent(file_paths):
    if not file_paths:
        return "No files uploaded.", None

    parsed_docs = {}
    for path in file_paths:
        # CORRECTED: Call the new function 'parse_document'
        content = parse_document(path)
        if content:
            parsed_docs[path] = content

    classified_docs = [classify_document(content) for content in parsed_docs.values()]
    print(f"Classified documents: {classified_docs}")
    
    checklist_report = verify_checklist(classified_docs)
    print("Checklist Report:")
    print(checklist_report)

    all_validation_reports = [rag_validate_document(path, content) for path, content in parsed_docs.items()]
    
    reviewed_doc_paths, json_report = generate_output_docs(list(parsed_docs.keys()), all_validation_reports)
    
    final_report = {
        **checklist_report, 
        **json_report
    }
    
    return reviewed_doc_paths, final_report

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_doc_path = os.path.join(project_root, "adgm_docs", "your_test_document.docx")
    
    if os.path.exists(test_doc_path):
        reviewed_doc_paths, final_report = run_corporate_agent([test_doc_path])
        print("\n--- Final Report ---")
        print(json.dumps(final_report, indent=2))
        print(f"Reviewed document paths: {reviewed_doc_paths}")
    else:
        print(f"Test document not found. Please place 'your_test_document.docx' in the 'adgm_docs' folder.")