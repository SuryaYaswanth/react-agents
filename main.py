import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializing components...")

embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GEMINI_API_KEY"),
                                           model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             api_key=os.getenv("GEMINI_API_KEY"))
vector_store = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),
                                   embedding=embeddings)

retriever = vector_store.as_retriever(
  search_kwargs={"k": 3}
  )

prompt_template = ChatPromptTemplate.from_template(
  """Answer the question based only on the follwing context:
    
    {context}

    Question: {question}

    Provide a detailed answer:"""
)

def format_docs(docs):
  """Format retrieved documents into a single string"""
  return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
  """
  Simple retrieval chain without LCEL.
  Manually retrieves documents, formats them, and generates a response.

  Limitations:
     - Manual step byb step execution
     - No built-in streaming support
     - No async support without additional code
     -Harder to compare with other chains
     - More verbose and error-prone
  """

  #Step 1: Retrieve relevant documents
  docs = retriever.invoke(query)

  #Step 2: Format retrieved documents into a single string
  context = format_docs(docs)

  #Step 3: Create prompt with context and question
  prompt = prompt_template.format(context=context, question=query)

  #Step 4: Generate response using LLM
  response = llm.invoke(input=prompt)

  return response.content

#=============================================
#IMPLEMENTATION 2: with LCEL (Langchain Expression Language)
#=============================================

def retrieval_chain_with_lcel():
  """
  Create a retrieval chain using LCEL (Langchain Expression Language).
  Returns a chain that can be invoked with ("question: "....")
  """

  retrieval_chain = (
    RunnablePassthrough.assign(
      context = itemgetter("question") |retriever | format_docs 
    ) 
    | prompt_template 
    | llm 
    | StrOutputParser()
  )

  return retrieval_chain

if __name__ == "__main__":
  print("Retrieving...")

  #Query
  query = "Explain RAG and its benefits in detail."

  #============================================
  # Option 0: Raw invocation without RAG
  # ===========================================
  print("\n"+"="*70)
  print("Option 0: Raw LLM invocation without RAG")
  response_raw = llm.invoke(input=query)
  print("\nResponse:")
  print(response_raw.content)



  #============================================
  # Option 1: Manual retrieval and generation (without LCEL)
  # ==========================================
  print("\n"+"="*70)
  print("Option 1: Manual retrieval and generation (without LCEL)")
  print("="*70)
  response_without_lcel = retrieval_chain_without_lcel(query)
  print("\nResponse:")
  print(response_without_lcel)     

  #============================================
  # Option 2: Retrieval chain using LCEL
  # ==========================================
  print("\n"+"="*70)
  print("Option 2: Retrieval chain using LCEL")
  print("="*70)
  retrieval_chain = retrieval_chain_with_lcel()
  response_with_lcel = retrieval_chain.invoke({"question": query})
  print("\nResponse:")
  print(response_with_lcel)                         
