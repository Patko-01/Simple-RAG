import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def print_doc_info(i, doc):
    print(f"\n\tDocument n.{i + 1}")
    print(f"\tSource: {doc.metadata['source']}")
    print(f"\tContent length: {len(doc.page_content)} characters")
    print(f"\tContent preview: {doc.page_content[:100].replace("\n", "\n\t")}...")
    print(f"\tmetadata: {doc.metadata}")

def load_docs(path="docs", print_all_docs=False):
    print(f"Loading documents from '{path}'")

    if not os.path.exists(path):
        print("Path does not exist!")
        return

    loader = DirectoryLoader(
        path=path,
        glob='*.txt',
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8-sig"}
    )

    documents = loader.load()

    if len(documents) == 0:
        print(f"No .txt files found in {path}")

    if print_all_docs:
        temp_docs = documents[:len(documents)]
    else:
        temp_docs = documents[:2]

    for i, doc in enumerate(temp_docs):
        print_doc_info(i, doc)

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("\nSplitting documents")

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n-----Chunk {i+1}-----")
            print(f"Source : {chunk.metadata['source']}")
            print(f"Length : {len(chunk.page_content)} characters")
            print(f"Content :")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5 : 
            print(f"... and  {len(chunks) - 5} more chunks.")       

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print(f"Creating embeddings and storing in DB:{persist_directory}")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating vector store... ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} #Algorithm for creating vector db from chunks
    )
    print("--- Finished creating vector store ---")
    print(f"Vector store saved to {persist_directory}")

    return vector_store

            

def main():
    print("\n######################\nSimple RAG starting...\n")

    #Loading the files
    documents = load_docs(path="docs", print_all_docs=True)

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks=chunks)



if __name__ == "__main__":
    main()