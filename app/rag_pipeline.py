import os
import requests
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import load_env
load_env()
os.environ['USER_AGENT'] = 'MyCorporateAgent/1.0'
api_key=os.env("OPENAI_API_KEY")
document_urls = [
    "https://assets.adgm.com/download/assets/adgm-ra-resolution-multipleincorporate-shareholders-LTDincorporationv2.docx/186a12846c3911efa4e6c6223862cd87",
    "https://www.adgm.com/documents/registration-authority/registration-and-incorporation/checklist/branch-nonfinancial-services-20231228.pdf",
    "https://www.adgm.com/documents/registration-authority/registration-and-incorporation/checklist/private-companylimited-by-guarantee-non-financialservices-20231228.pdf",
    "https://assets.adgm.com/download/assets/ADGM+Standard+Employment+Contract+Template+-+ER+2024+(Feb+2025).docx/ee14b252edbe11efa63b12b3a30e5e3a",
    "https://assets.adgm.com/download/assets/ADGM+Standard+Employment+Contract+-+ER+2019+-+Short+Version+(May+2024).docx/33b57a92ecfe11ef97a536cc36767ef8",
    "https://www.adgm.com/documents/office-of-data-protection/templates/adgm-dpr-2021-appropriate-policy-document.pdf",
    "https://assets.adgm.com/download/assets/Templates_SHReso_AmendmentArticles-v1-20220107.docx/97120d7c5af911efae4b1e183375c0b2?forcedownload=1",
]

web_urls = [
    "https://www.adgm.com/registration-authority/registration-and-incorporation",
    "https://www.adgm.com/setting-up",
    "https://www.adgm.com/legal-framework/guidance-and-policy-statements",
    "https://www.adgm.com/operating-in-adgm/obligations-of-adgm-registered-entities/annual-filings/annual-accounts",
    "https://www.adgm.com/operating-in-adgm/post-registration-services/letters-and-permits",
    "https://en.adgm.thomsonreuters.com/rulebook/7-company-incorporation-package",
]

def build_vector_store(project_root):
  
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    os.makedirs(os.path.join(project_root, "adgm_docs"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "adgm_vectorstore"), exist_ok=True)
    
    def download_file(url, folder):
        file_name = url.split('/')[-1].split('?')[0]
        file_path = os.path.join(folder, file_name)
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    
    local_docs = []
    for url in document_urls:
        try:
            file_path = download_file(url, os.path.join(project_root, "adgm_docs"))
            if file_path.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                continue
            local_docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    web_loader = WebBaseLoader(web_urls)
    web_docs = web_loader.load()
    docs = web_docs + local_docs

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)
    vectorstore.save_local(os.path.join(project_root, "adgm_vectorstore"))
    print("Vector store created and saved successfully.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_vector_store(project_root)