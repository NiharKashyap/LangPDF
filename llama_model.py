from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("data/", glob="**/*.pdf")
books = loader.load()
len(books)


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(books)


from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model="llama3", show_progress=True),
    persist_directory="./chroma_db",
)


question = "Who is Zahra?"
docs = vectorstore.similarity_search(question)

print(docs)



from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# loading the vectorstore
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OllamaEmbeddings(model="llama3"))

# loading the Llama3 model
llm = Ollama(model="llama3")

# using the vectorstore as the retriever
retriever = vectorstore.as_retriever()

# formating the docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# loading the QA chain from langchain hub
rag_prompt = hub.pull("rlm/rag-prompt")

# creating the QA chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# running the QA chain in a loop until the user types "exit"
while True:
    question = input("Question: ")
    if question.lower() == "exit":
        break
    answer = qa_chain.invoke(question)

    print(f"\nAnswer: {answer}\n")



