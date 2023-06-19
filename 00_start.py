from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
import os

import openai
# from langchain.embeddings import OpenAIEmbeddings
# https://www.youtube.com/watch?v=DYOU_Z0hAwo&list=PLgJWF7IEIkmJa3JhjAOvMQF1u3aCw2FmP&index=4&ab_channel=SamuelChan
load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv('API_BASE')
openai.api_version = "2022-12-01"
openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
text = "Algoritma is a data science school based in Indonesia and Supertype is a data science consultancy with a distributed team of data and analytics engineers."
doc_embeddings = embeddings.embed_documents([text])


print(doc_embeddings)




# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# TAL_TEST = os.getenv('TAL_TEST')
# print(OPENAI_API_KEY)
# print(TAL_TEST)
