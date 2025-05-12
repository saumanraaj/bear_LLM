import os
from pathlib import Path
import faiss
from filtered_loader import load_clean_text_from_pdf

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk--"  

# Load PDFs
pdf_paths = [
    "./papers/lab1.pdf",
    "./papers/lab2.pdf",
    "./papers/Superlative mechanical energy absorbing efficiency discovered through self-driving lab-human partnership (1).pdf",
    "./papers/A Bayesian experimental autonomous researcherfor mechanical design (1).pdf"
]

print("Loading and processing PDF documents...")
pdf_documents = [load_clean_text_from_pdf(path, max_pages=30) for path in pdf_paths]
print(f"Loaded {len(pdf_documents)} documents from papers.")

# Function to load GitHub files
def load_codebase_as_documents(folder, exts=[".py", ".md", ".txt"]):
    code_docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix in exts:
            try:
                text = path.read_text(encoding="utf-8")
                if len(text.strip()) > 50:
                    code_docs.append(Document(text=text, metadata={"source": str(path)}))
            except Exception as e:
                print(f"Skipped {path.name}: {e}")
    return code_docs

# Load GitHub repo
print("Loading GitHub repository files...")
code_documents = load_codebase_as_documents("./gcs")
print(f"Loaded {len(code_documents)} documents from GitHub repo.")
for doc in code_documents[:3]:
    print(f"- {doc.metadata['source']}")

# Combine PDFs and code
documents = pdf_documents + code_documents

# Parse into nodes
parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)
print(f"Parsed into {len(nodes)} text chunks.")

# Embed and store in FAISS
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
faiss_index = faiss.IndexFlatL2(1536)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents=documents,
    embed_model=embed_model,
    storage_context=storage_context
)
print("Vector index built.")

# Setup the GPT-4 Turbo query engine
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4-turbo", temperature=0.2),
    similarity_top_k=3
)
print("Query engine ready. You can now ask questions.")

# Start interactive loop
try:
    while True:
        question = input("\nAsk something (or type 'exit'): ")
        if question.lower() == "exit":
            print("Exiting...")
            break
        response = query_engine.query(question)
        print("\nAnswer:")
        print(response)
except Exception as e:
    print(f"Error: {e}")
