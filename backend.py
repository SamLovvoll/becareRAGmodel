from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from datasets import Dataset
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4
from werkzeug.utils import secure_filename
import json

from flask import Flask
app = Flask(__name__)

class backend:
    # variables (I'm pretty sure I don't have to initialize these as empty but it helps give some extra information)

    # path to json file (change this to where there is a json file)
    namesPath = ""
    # the LangChain RAG model
    qa = None
    tokenizer = None
    # name of the index in pinecone
    index_name = 'langchain-retrieval-augmentation-stroke'
    # index (the vector database)
    index = None
    # text embedder
    embed = None
    # llm used
    llm = None

    # user functions

    # returns all the documents currently in the the database
    def getDocs(self):
        # load the JSON file and convert to a dictionary
        dict = json.load(open(self.namesPath))

        # check if there's an item called files (create one if not)
        if 'files' not in dict:
            # make it there!
            dict['files'] = []

        # return the list
        return dict['files']

    # querys the RAG model and returns the answer, along with source documents in a dict. keys: "result" and "source_documents"
    def query(self, query):
        return self.qa({"query": query})

    # with a txt file (Werkzeug FileStorage object) as a parameter, uploads the file to the database (returns True/False success)
    def uploadFile(self, file):
        # check the filetype and confirm it is in text form
        if file.mimetype[:4] != 'text':
            # app.logger.debug(file.mimetype[:4])
            return False

        # parse the file into text data
        input_data = file.stream.read().decode("utf-8")

        # get the filename and make it secure (also remove the .txt because every file has that)
        filename = secure_filename(file.filename)[:-4]

        # add the file to the list of files (if duplicates exist, will return modified filename)
        filename = self.addFileToJson(filename)

        # create a dictionary to convert into a dataset (huggingface does not accept string as input data from memory)
        tempDict = {}
        tempDict.__setitem__('text', [input_data])

        # convert the dictionary to a dataset we can then chunk and add to our database
        dataset = Dataset.from_dict(tempDict, split='train')

        # load the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, # how many tokens per chunk (max, not average)
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""] # what chunks can be split by
        )

        # we are going to vectorize these chunks in batch sizes of batch_limit or more
        batch_limit = 100

        texts = []
        metadatas = []

        for i, record in enumerate(tqdm(dataset)):
            # first get metadata fields for this record
            metadata = {
                'filename': filename
            }
            # now we create chunks from the record text
            record_texts = text_splitter.split_text(record['text'])
            # create individual metadata dicts for each chunk
            record_metadatas = [{
                "chunk": j, "text": text, **metadata
            } for j, text in enumerate(record_texts)]
            # append these to current batches
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            # if we have reached the batch_limit we can add texts
            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = self.embed.embed_documents(texts)
                self.index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []

        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = self.embed.embed_documents(texts)
            self.index.upsert(vectors=zip(ids, embeds, metadatas))

        # successful upload
        return True

    # specify the name of the document you want to delete (with no file extension), and it will be removed from the database
    def deleteDoc(self, docName):
        # remove file from json document
        self.removeFileFromJson(docName)

        # delete (Pinecone makes this easy)
        self.index.delete(
            filter={
                "filename": docName
            }
        )

    # will delete all documents from database (implement last)
    def deleteAllDocs(self):
        # delete the index
        pinecone.delete_index(self.index_name)

        # make new index, now empty
        pinecone.create_index(
            name=self.index_name,
            metric='cosine',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

        # connect to the index/re-setup qa
        self.setQA()

        # clear the json file
        dict = {'files': []}
        with open(self.namesPath, "w") as outfile:
            json.dump(dict, outfile)

    # helper methods

    # connects to all APIs, sets up the model and initializes everything
    def __init__(self, jsonPath, pinecone_API_key, openai_API_key, pinecone_environment = 'us-east1-gcp'):
        # store the path to the json file
        self.namesPath = jsonPath

        # set up the tokenizer
        global tokenizer
        tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.tokenizer = tiktoken.get_encoding('cl100k_base')

        # connect to all APIs

        # set up proxy server so we can use pinecone through pythonanywhere (the python hosting service I use)
        openapi_config = OpenApiConfiguration.get_default_copy()
        openapi_config.proxy = "http://proxy.server:3128"

        # connect to the Pinecone API
        pinecone.init(
            api_key=pinecone_API_key,
            environment=pinecone_environment,
            openapi_config=openapi_config
        )

        # connect to openAI API for embedding
        model_name = 'text-embedding-ada-002'

        self.embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_API_key
        )

        # create llm to be used in the chain
        self.llm = ChatOpenAI(
            openai_api_key=openai_API_key,
            model_name='gpt-3.5-turbo',
            temperature=0.0
        )

        # connect and generate the qa model
        self.setQA()


    # removes the file name from the json
    def removeFileFromJson(self, docName):
        # verify that docName is in documents
        files = self.getDocs()

        if docName not in files:
            return False

        # remove the desired file from the list
        files.remove(docName)

        # put into dictionary
        dict = {'files': files}

        # send dictionary back to json file
        with open(self.namesPath, "w") as outfile:
            json.dump(dict, outfile)

    # adds a file name to the json and returns the file name (possibly modified for the case of duplicates present)
    def addFileToJson(self, docName):
        # load the current info in the json file
        files = self.getDocs()

        # check for duplicate docname
        if docName in files:
            # keep on trying to add docname(1) or docname(2) until no duplicate is found
            num=1
            while (docName + "({n})".format(n=num)) in files:
                num=num+1

            docName = docName + "({n})".format(n=num)

        # append new docname to list
        files.append(docName)

        # put into dictionary
        dict = {'files': files}

        # send dictionary back to json file
        with open(self.namesPath, "w") as outfile:
            json.dump(dict, outfile)

        return docName

    # token length function
    def tiktoken_len(self, text):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    # connects to the Pinecone index and creates a LangChain chain to be used for querying
    def setQA(self):
        # if index does not exist, create it
        if self.index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=self.index_name,
                metric='cosine',
                dimension=1536  # 1536 dim of text-embedding-ada-002
            )

        # connect to the index and set it
        self.index = pinecone.Index(self.index_name)

        # create a vectorstore (retriever) using pinecone
        text_field = "text"

        vectorstore = Pinecone(
            self.index, self.embed.embed_query, text_field
        )

        # use langchain to make a chain to do all the hard work for us

        # set the chain used
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )