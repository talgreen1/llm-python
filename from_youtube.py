from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import openai
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

load_dotenv()
# https://www.youtube.com/watch?v=U0sJ3uVtylM&ab_channel=ShwetaLodha


openai.api_type = "azure"
openai.api_base = os.getenv('API_BASE')
openai.api_version = "2022-12-01"
openai.api_key = os.getenv('OPENAI_API_KEY')

with open('./news/result.txt') as f:
    text = f.read()

text_splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=0)
text = text_splitter.split_text(text)

print(text)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(text, embeddings)
chain = RetrievalQA.from_chain_type(
    llm=AzureOpenAI(model_kargs={'engine': 'text-davinci-002'}), retriver=vectorstore.as_retriever(),
                    chain_type='stuff')

query = "What are the effects of legislations surrounding emissions on the Australia coal market?"
res = chain.run(query)
print(res)
