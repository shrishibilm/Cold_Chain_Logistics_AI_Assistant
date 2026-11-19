import os
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import pypdf

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

# Suppress the Windows symlink warning noise completely
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

script_dir =Path(__file__).resolve().parent
project_root = script_dir.parent
load_dotenv(project_root / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set. Please set it in your .env file.")

# Set up clean production data cache directory
cache_dir = project_root / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
HASH_CACHE_FILE = cache_dir / "hash_cache.json"

hash_cache = {}

if HASH_CACHE_FILE.exists():
    try:
        with open(HASH_CACHE_FILE, "r") as f:
            hash_cache = json.load(f) # if the file exists, load the hash cache from it. load means to read the file and parse the JSON content into a Python dictionary. 

    except Exception as e:
        print(f"Error loading hash cache: {e}. Starting with an empty cache.")
        hash_cache = {}


# ==============================================================================
# 2. DYNAMIC EMBEDDINGS MODEL SELECTION
# ==============================================================================

EMBEDDINGS_MODEL_SETTING = os.getenv("embeddings_model", "LOCAL").strip().upper() 

if EMBEDDINGS_MODEL_SETTING == "OPENAI":
    print("🌐 Mode: Utilizing Cloud OpenAI Embeddings (1536 Dim)...")
    embeddings = OpenAIEmbeddings()
    INDEX_NAME = "fde-sop-index-openai"  # Isolated OpenAI Index
    TARGET_DIMENSION = 1536
else:
    # Read the explicit model identifier casing string from the .env parameters
    local_model_target = os.getenv("Local_Embedding_Model", "BAAI/bge-m3").strip()

    print(f"🏠 Mode: Local Fallback Settings Activated. Launching [{local_model_target}]")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_target,  # Passes parameter dynamically
        model_kwargs={'device': 'cpu'}
    )
    INDEX_NAME = "fde-sop-index-local"  # Isolated Local Model Index
    TARGET_DIMENSION = 1024  # Standard width for BGE-M3

# ==============================================================================
# 3. PINECONE PROVISIONING
# ==============================================================================
print(f"Connecting to Pinecone Index Target: [{INDEX_NAME}]...")
pc = Pinecone(api_key=PINECONE_API_KEY) 

existing_indexes = pc.list_indexes().names()
print(f"Existing Indexes: {existing_indexes}") 

# Self-healing verification in case an index was created with a wrong legacy dimension
if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != TARGET_DIMENSION:
        print(f"⚠️ Fixing tracking: Purging mismatched {desc.dimension} dim index...")
        pc.delete_index(INDEX_NAME)
        existing_indexes = [name for name in existing_indexes if name != INDEX_NAME] # here we are filtering out the INDEX_NAME from existing_indexes if it was deleted due to dimension mismatch. why do we need this step is because we want to ensure that the list of existing indexes accurately reflects the current state of the Pinecone environment after the deletion. If we didn't update the list, it would still contain the name of the deleted index, which could lead to confusion or errors in subsequent operations that rely on this list.

if INDEX_NAME not in existing_indexes:
    print(f"Creating isolated target index: {INDEX_NAME} ({TARGET_DIMENSION} Dim)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=TARGET_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index_client = pc.Index(INDEX_NAME) 

print(f"✅ Connected to Pinecone Index: {INDEX_NAME} ({TARGET_DIMENSION} Dim)")

vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings) 

print(f"✅ Pinecone Vector Store ready for ingestion into: {INDEX_NAME} ({TARGET_DIMENSION} Dim)")


# ==========================================
# 4. ROBUST POLYMORPHIC PARSER
# ==========================================

# so no matter what the file is we want to first convert it into a list of langchain document objects and then convert them into chunks using text splitter?
# Every supported file type in your parser ends up as a list of LangChain Document objects. How you get there depends on the file type.

# File  type	What the code does
# .txt	Creates one Document for the whole file, then splits it into chunks.
# .md	Creates Document objects by heading section, then splits long sections into smaller chunks.
# .pdf	Creates one Document per page with extracted text, then splits long pages into smaller chunks.
# .csv / .xlsx	Creates one Document per non-empty row. Each row is already a chunk, so it skips the text splitter.

def parse_and_chunk_document(doc_path: Path) -> list[Document]:
    ext = doc_path.suffix.lower()
    raw_chunks: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    
    if ext == ".md":
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")] # tuple("markdown marker","Metadata key")
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_text = doc_path.read_text(encoding="utf-8") # what this returns is a string containing the entire content of the markdown file located at doc_path. The read_text method reads the file and decodes it using UTF-8 encoding, returning the text as a single string. If the file is large, this could consume a significant amount of memory, but for typical markdown files, it should be manageable.
        header_docs = md_splitter.split_text(raw_text) # What this returns is lis of langchain document objects, where each document corresponds to a section of the markdown file that was split based on the specified headers. Each document in the list contains page con
        raw_chunks = text_splitter.split_documents(header_docs)

    elif ext == ".txt":
        raw_text = doc_path.read_text(encoding="utf-8")
        raw_docs = [Document(page_content=raw_text)] # here the raw text becomes a single Document object, which is then wrapped in a list to create a collection of documents. This is necessary because the split_documents method expects a collection of Document objects as input, even if there's only one document in this case.
        raw_chunks = text_splitter.split_documents(raw_docs) #split_documents() takes a collection of LangChain Document objects and returns a list of smaller Document objects.

    elif ext == ".pdf":
        pdf_docs = []
        try:
            with open(doc_path, "rb") as f:
                reader = pypdf.PdfReader(f) # after reading it becomes a PdfReader object, which provides access to the contents of the PDF file, including its pages, metadata, and other information. The PdfReader class is part of the PyPDF2 library, which is used for working with PDF files in Python.
                for page_num, page in enumerate(reader.pages): # reader.pages is a list-like objects of all the pages of the PDF document. The enumerate function is used to iterate over this list, providing both the index (page_num) and the page object (page) for each iteration. This allows you to access each page's content and its corresponding page number in a single loop.
                    page_text = page.extract_text() # Extracts only texts from the page, ignoring images, annotations, and other non-text elements. The extracted text is returned as a string. If the page has no text, it may return None or an empty string.
                    if page_text and page_text.strip(): # The condition checks if page_text is not None and contains non-whitespace characters. The strip() method removes any leading and trailing whitespace from the string, and if the resulting string is not empty, it evaluates to True. This ensures that only pages with actual text content are processed further.
                        pdf_docs.append(Document(page_content=page_text, metadata={"page_number": page_num + 1}))
            raw_chunks = text_splitter.split_documents(pdf_docs)
        except Exception as e:
            print(f"  ❌ Error parsing PDF {doc_path.name}: {e}")
            return []

    elif ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(doc_path) if ext == ".csv" else pd.read_excel(doc_path)
        except Exception as e:
            print(f"  ❌ Error reading table: {e}")
            return []
            
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_items = [
                f"{str(col)}: {str(val)}" 
                for col, val in row_dict.items() 
                if pd.notna(val) and str(val).strip() != ""
            ]
            
            if row_items:
                row_text = " | ".join(row_items)
                doc_item = Document(page_content=row_text, metadata={"row_index": int(idx)})
                raw_chunks.append(doc_item)

    # sanity check
    valid_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.strip()
        if clean_text:
            chunk.page_content = clean_text
            valid_chunks.append(chunk)
            
    return valid_chunks

# ==========================================
# 5. INCREMENTAL PIPELINE WITH BATCHING
# ==========================================

policy_dir = project_root / "data" / "policy"
target_patterns = ["*.md", "*.txt", "*.pdf", "*.csv", "*.xlsx"]
current_files = {}
for pattern in target_patterns:
    for file_path in policy_dir.glob(pattern):
        current_files[file_path.name] = file_path


print(f"Found {len(current_files)} policy file(s) in {policy_dir}...")

updated_cache = {}
cache_modified = False

cached_filenames = set(hash_cache.keys()) # why putting keys inside set is because we want to compare the file names in the hash cache with the current files in the policy directory. By converting the keys of the hash_cache dictionary into a set, we can easily perform set operations to identify which files have been deleted or added since the last run of the script. Sets provide efficient membership testing and allow us to quickly determine differences between the cached file names and the current file names.
current_filenames = set(current_files.keys())
deleted_files = cached_filenames - current_filenames # compares the file names and stores files available in cached but not in the current files. This is useful for identifying files that have been removed from the policy directory since the last run of the script.

for deleted_file in deleted_files:
    print(f"🗑️ Detected deleted file: {deleted_file}. Purging from Pinecone...")
    try:
        index_client.delete(filter={"source_file": {"$eq": deleted_file}}) # deleting all the chunks with that deleted file name metadata from the Pinecone index. The filter specifies that only documents with a metadata field source_file equal to the deleted file name should be removed. This ensures that only the relevant documents associated with the deleted file are purged from the index, while leaving other documents intact.
        cache_modified = True # this flag indicates that the hash cache has been modified due to the deletion of a file, which will trigger an update to the hash cache file later in the script.
    except Exception as e:
        print(f"  ❌ Failed to purge {deleted_file}: {e}")

for file_name, file_path in current_files.items():
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    updated_cache[file_name] = file_hash
    
    if hash_cache.get(file_name) == file_hash:
        print(f"✨ Skipped (Unchanged): {file_name}")
        continue
        
    print(f"🔄 Processing updates: {file_name}...")
    cache_modified = True

    try:
        try:
            index_client.delete(filter={"source_file": {"$eq": file_name}})
        except Exception:
            pass 
        
        chunks = parse_and_chunk_document(file_path)
        if not chunks: # meaning 
            print(f"  ⚠️ No valid text chunks extracted from {file_name}.")
            continue
       
        explicit_ids = []
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = file_name
            chunk.metadata["file_format"] = file_path.suffix.replace(".", "").upper()
            chunk.metadata["document_type"] = "Compliance Asset"
            explicit_ids.append(f"{file_name}-chunk-{idx}")
            
        batch_size = 100
        total_chunks = len(chunks)
        print(f"  📤 Upserting {total_chunks} chunk(s) in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch_docs = chunks[i : i + batch_size]
            batch_ids = explicit_ids[i : i + batch_size]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    except Exception as e:
        print(f"❌ Error during ingestion of {file_name}: {e}")
        updated_cache.pop(file_name, None)

    
# ==========================================
# 6. SYNC HASH CACHE
# ==========================================
if cache_modified: # meaning if any file was added, updated, or deleted, we need to update the hash cache file to reflect the current state of the policy directory. This ensures that the next time the script runs, it can accurately determine which files have changed and need to be reprocessed.
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(updated_cache, f, indent=4) # here we are writing the updated_cache dictionary to the hash cache file in JSON format. The json.dump function serializes the dictionary into a JSON string and writes it to the file. The indent=4 argument is used to format the JSON output with an indentation of 4 spaces, making it more human-readable. This step ensures that the hash cache file reflects the current state of the policy directory after any changes have been processed.
    print("✅ Ingestion & cache update complete.")
else:
    print("🌴 Index is already up-to-date.")