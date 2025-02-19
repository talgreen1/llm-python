from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
import os
import nltk

# nltk.download()
load_dotenv()
# https://www.youtube.com/watch?v=U0sJ3uVtylM&ab_channel=ShwetaLodha
import openai
openai.api_type = "azure"
openai.api_base = os.getenv('API_BASE')
openai.api_version = "2022-12-01"
openai.api_key = os.getenv('OPENAI_API_KEY')


embeddings = OpenAIEmbeddings(chunk_size=1)

# loader = TextLoader('news/summary.txt')
# loader = DirectoryLoader('news', glob="**/*.txt")
loader = DirectoryLoader('news', glob="**/result.txt")

documents = loader.load()
print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# print(texts)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    # llm=OpenAI(),
    llm=AzureOpenAI(model_kwargs={'engine':'text-davinci-002'}),
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)

def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

query("What are the effects of legislations surrounding emissions on the Australia coal market?")
query("What are China's plans with renewable energy?")
query("Is there an export ban on Coal in Indonesia? Why?")
query("Who are the main exporters of Coal to China? What is the role of Indonesia in this?")