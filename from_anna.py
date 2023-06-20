import os
import openai
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding, GPTVectorStoreIndex, SimpleWebPageReader, ServiceContext, load_index_from_storage, StorageContext
from llama_index.node_parser import SimpleNodeParser
import pathlib
import winsound
load_dotenv()
deployment_name = 'gpt35'
openai.api_type = "azure"
openai.api_base = "https://aicontesteuteams.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Engine = deployment_name

# class AddingDataToGPT:
class AddingDataToGPT():
    def __init__(self):
        self.index = None
        self.persist_dir = "./storage"


        self.load_models()
        if os.path.exists(self.persist_dir):
            self.read_from_storage()
        else:
            self.build_storage('internal_links_file')

        # Query index with a question
        query_engine = self.index.as_query_engine()
        # "what is remote-dx"

        # response = query_engine.query("how to connect to hypervizor?")
        # print(response)

        # chat_engine = self.index.as_chat_engine(
        #  chat_mode='condense_question',
        #  verbose=True, service_context = self.service_context)
        # chat_engine.chat_repl()


        def load_models(self):
            # Create LLM via Azure OpenAI Service
            llm = AzureOpenAI(model_name=deployment_name, deployment_name=deployment_name, model_kwargs={
                "api_key": openai.api_key,
                "api_base": openai.api_base,
                "api_type": openai.api_type,
                "api_version": openai.api_version,
            })


            embedding_llm = LangchainEmbedding(OpenAIEmbeddings(model='text-embedding-ada-002', chunk_size=1))

            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embedding_llm
            )

            self.service_context = service_context


        def read_internal_links(self, file_name):
            working_directory = os.getcwd()
            file_path = os.path.join(working_directory, file_name)


        with open(file_path) as f:
            urls_list = f.readlines()

        # remove new line characters
        urls_list = [x.strip() for x in urls_list]
        return urls_list


        def create_indexes(self, urls_list):
            documents = SimpleWebPageReader(html_to_text=True).load_data(urls_list)
            index = GPTVectorStoreIndex.from_documents(documents=documents, service_context=self.service_context)
            index.storage_context.persist()
            return index


        def create_indexes2(self, urls_list):
            documents = SimpleWebPageReader(html_to_text=True).load_data(urls_list)
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(documents)
            index = GPTVectorStoreIndex(nodes=nodes, service_context=self.service_context)
            index.storage_context.persist()
            return index



        def read_from_storage(self):
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            # load index
            self.index = load_index_from_storage(storage_context=storage_context, service_context=self.service_context)



        def build_storage(self, links_file_path):
            urls_list = self.read_internal_links(links_file_path)
            self.index = self.create_indexes2(urls_list)



        def ask(self, enquiry):
            query_engine = self.index.as_query_engine()
            response = query_engine.query(enquiry)
            return response



        def exit_chat(self, val):
            is_quit = val.lower() == "exit" or val.lower() == "x" or val.lower() == "quit" or val.lower() == "q"
            return is_quit



        def get_user_input(self):
            print("")
            print("")
            print("=================================================")
            return input("Enter your query: ")


################################################################################################
###  main  Q and A  loop
# ##############################################################################################

chatbot = AddingDataToGPT()
query = chatbot.get_user_input()

while chatbot.exit_chat(query) == False:
    print("")

    response = chatbot.ask(query)
    winsound.Beep(200, 500)
    print(response)

    query = chatbot.get_user_input()

winsound.Beep(700, 200)
