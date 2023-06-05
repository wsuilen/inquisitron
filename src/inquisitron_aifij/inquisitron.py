import os

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo, \
    create_python_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.tools import HumanInputRun
from langchain.tools import Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import Chroma

import pandas as pd


class Inquisitron:
    def __init__(self, openai_api_key, serpapi_api_key, csv_file_path, llm_temperature=0, column_names=None, ):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["SERPAPI_API_KEY"] = serpapi_api_key
        self.csv_file_path = csv_file_path
        self.llm_temperature = llm_temperature
        self.column_names = column_names

        self.llm = self._init_llm()
        self.openai_embeddings = self._init_embeddings()
        self.df = self._set_df()
        self.data = self._import_data()
        self.vector_store = self._set_vector_store()
        self.memory = self._init_memory()

        self.agents = self._init_agents()
        self.tools = self._init_tools()
        self.agent = self._init_agent()

    def _init_llm(self):
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=self.llm_temperature)

    def _init_embeddings(self):
        return OpenAIEmbeddings()

    def _set_df(self):
        return pd.read_csv(self.csv_file_path)

    def _import_data(self):
        if self.column_names is not None:
            loader = DataFrameLoader(self.df, page_content_column=self.column_names)
        else:
            loader = DataFrameLoader(self.df)
        return loader.load()

    def _set_vector_store(self):
        if self.vector_store is None:
            try:
                vector_store = Chroma(
                    persist_directory="vectorstore",
                    embedding_function=self.openai_embeddings)
            except FileNotFoundError:
                print("No vectorstore found, creating new one.")
                vector_store = Chroma.from_documents(
                    documents=self.data,
                    embedding=self.openai_embeddings,
                    persist_directory="vector_store",
                )
            return vector_store

    def _init_memory(self):
        return ConversationBufferMemory(memory_key="chat_history")

    def _get_input(self) -> str:
        print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
        contents = []
        while True:
            try:
                line = input()

            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)

    def _init_agents(self):
        search = SerpAPIWrapper()

        python_repl = create_python_agent(llm=self.llm,
                                          tool=PythonREPLTool(),
                                          verbose=True)

        metadata_search = RetrievalQA.from_chain_type(llm=self.llm,
                                                      chain_type="stuff",
                                                      retriever=self.vector_store.as_retriever())

        human_input = HumanInputRun(input_func=self._get_input)

        panda_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), self.df, verbose=True)

        vectorstore_info = VectorStoreInfo(
            name="...",  # Name the input data
            description="...",  # Give a description of the input data
            vectorstore=self.vector_store
        )
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
        agent_executor = create_vectorstore_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=True
        )

        agents = {
            "search": search,
            "python_repl": python_repl,
            "metadata_search": metadata_search,
            "human_input": human_input,
            "panda_agent": panda_agent,
            "agent_executor": agent_executor
        }

        return agents

    def _init_tools(self):
        tools = [
            Tool(
                name="Searching the web",
                func=self.tools["search"].run,
                description="useful for when you need to answer questions about current events or the current state of the world"
            ),
            Tool(
                name="python code generator",
                func=self.tools["python_repl"].run,
                description="useful when a python script needs to be run in order to answer a question related to calculation"
            ),
            Tool.from_function(
                name="Data QA Tool",
                func=self.tools["metadata_search"].run,
                description="useful for when you need to answer simple questions about the emedded csv file. ."
            ),
            Tool.from_function(
                name="Human clarifier",
                func=self.tools["human_input"],
                description="Useful when you need more information from the user for a next step."
            ),
            Tool(
                name="Pandas Agent",
                func=self.tools["panda_agent"].run,
                description="useful when gathering information about a pandas dataframe"
            ),
            Tool(
                name="Vectorstore Agent",
                func=self.tools["agent_executor"].run,
                description="Always use this! The vectorstore contains all the information about the sexworker ads!"
            ),
            #     It is possible to add more tools here. Langchain provides many tools and toolkits
            #     out of the box: https://python.langchain.com/en/latest/modules/agents/toolkits.html
        ]

        return tools

    def _init_agent(self):
        return initialize_agent(self.tools,
                                self.llm,
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                verbose=True,
                                memory=self.memory,
                                handle_parsing_errors=True)

    def run(self, query):
        self.agent.run(query)
