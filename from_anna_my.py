import os
import openai
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding, GPTVectorStoreIndex, SimpleWebPageReader, ServiceContext
from langchain.document_loaders import DirectoryLoader, TextLoader


load_dotenv()
deployment_name = 'gpt35'
openai.api_type = "azure"
# openai.api_base = "https://aicontesteuteams.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
# openai.api_key = os.getenv("OPENAI_API_KEY")



openai.api_type = "azure"
openai.api_base = os.getenv('API_BASE')
openai.api_version = "2022-12-01"
openai.api_key = os.getenv('OPENAI_API_KEY')




# openai.Engine = deployment_name


llm = AzureOpenAI(model_name=deployment_name, deployment_name=deployment_name, model_kwargs={
                "api_key": openai.api_key,
                "api_base": openai.api_base,
                "api_type": openai.api_type,
                "api_version": openai.api_version,
            })

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(model='text-embedding-ada-002', chunk_size=1))

loader = DirectoryLoader('news', glob="**/result.txt")

documents = loader.load()
print(len(documents))