from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

path='./man2'

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False,
)

#read all the documents in the directory
def loadDocs(path:str):
    docs = DirectoryLoader(path,glob='*.man',use_multithreading=False,show_progress=True)
    return docs.load()

#convert to vector
def convertToVector(docs):
    return Chroma.from_documents(docs,embeddings,persist_directory='chroma')

docs=loadDocs(path=path)
print("LOADING DONE")
chunks = text_splitter.split_documents(docs)
convertToVector(chunks)
print("done bro")