from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os

from dotenv import load_dotenv


load_dotenv()

# Step 1: Upload the document to the loader to convert into langchain documents
def document_loader(docs_path):
    """upload the txt file and convert it into langchain documents"""
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt", # only look for .txt files
        loader_cls = TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents")
    print(len(documents))

    for i, doc in enumerate(documents[:4], 1):
        print(f"Document: {i}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content preview: {doc.page_content[:200]}...")
        print("============================================")
    
    return documents

# Step 2: Use splitter to perform the chunks
def chunk_split(documents):

    text_splitter = CharacterTextSplitter(
                    chunk_size = 1000,
                    chunk_overlap = 0
                   )
    chunks = text_splitter.split_documents(documents=documents)
    num_chunks = len(chunks)
    print("Number of chunks:", num_chunks)



    if chunks:
        for i, chunk in enumerate(chunks[:4], 1):
            print(f"Chunk: {i}")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Content length of chunk: {len(chunk.page_content)} characters")
            print(f"Content preview: {chunk.page_content[:100]}...")
    
    return chunks


documents = document_loader("./docs")
chunks = chunk_split(documents)

# Step 3: Save the retrieved chunks in the vector store

def create_vector_store(chunks, persist_directory):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("--- Creating vector store ---")
    db = Chroma.from_documents(
          documents=chunks,
          embedding=embedding,
          persist_directory=persist_directory,
          collection_metadata= {"hsnw:space": "cosine"} # algorithhm to be cosine similarity
          ) 
    print("--- Finished Creating Vector Store ---")

    print(f"Vectore store saved in {persist_directory}")
    
    return db

def main():
    """Main ingestion pipeline"""
    print(" Starting RAG Ingestion pipeline")
    
    # define path
    docs_path = "./docs"
    persist_directory='db/chroma_basic_db'

    # check if vector store already exists
    if os.path.exists(persist_directory):
        print(f"The vector store already exists")

        #load the existing vector store
        db = Chroma(embedding_function=embedding,persist_directory=persist_directory,
                    collection_metadata={"hsnw:space": "cosine"})

    else:
        print(f"create initializing the vector store")    

        # Load the file
        documents = document_loader(docs_path=docs_path)

        # Chunk the documents
        chunks = chunk_split(documents)

        # create the embedings and store in the DB
        db = create_vector_store(chunks=chunks, persist_directory=persist_directory)
    
    print(f"Ingestion complete! You documents are now ready for RAG queries")
    return db


if __name__=="__main__":
    main()
