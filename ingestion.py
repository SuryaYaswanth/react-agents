from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
load_dotenv()



if __name__ == "__main__":
    print("Ingesting.....")
    loader = TextLoader("./mediumblog.txt",encoding="utf-8")
    document = loader.load()

    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"Number of chunks: {len(texts)}")

    embeddings = GoogleGenerativeAIEmbeddings(gemini_api_key=os.getenv("GEMINI_API_KEY"),
                                              model="gemini-embedding-001")

    print("Ingesting to Pinecone....")
    PineconeVectorStore.from_documents(texts,
                                       embeddings,
                                       index_name=os.getenv("INDEX_NAME")
                                       )
    print("Ingestion completed!")