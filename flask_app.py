# this code is mostly from https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/jumpstart-foundation-models/question_answering_retrieval_augmented_generation/question_answering_langchain_jumpstart.html
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from decouple import config
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from flask import Flask, request
from datasets import Dataset
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# file location of JSON file named filenames.json
namesPath = "/home/Slovvoll/mysite/filenames.json"

# prompts the user to upload a file OR if file is detected, uploads it to the pinecone database
@app.route("/updateDataset", methods=["GET", "POST"])
def download():
    # append this to the normal site text to indicate data was uploaded
    success = ''

    # if file is detected: chunk & upload data to pinecone database
    if request.method == "POST":
        # get the uploaded text file
        input_file = request.files["input_file"]
        input_data = input_file.stream.read().decode("utf-8")

        # get the filename and make it secure (also remove the .txt because every file has that)
        filename = secure_filename(input_file.filename)[:-4]

        # add the file to the list of files (if duplicates exist, will return modified filename)
        filename = addFileToJSON(filename)

        # create a dictionary to convert into a dataset (huggingface does not accept string as input data from memory)
        tempDict = {}
        tempDict.__setitem__('text', [input_data])

        # convert the dictionary to a dataset we can then chunk and add to our database
        dataset = Dataset.from_dict(tempDict, split='train')

        # load the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, # how many tokens per chunk (max, not average)
            chunk_overlap=20,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""] # what chunks can be split by
        )

        # connect to the index
        index = connectToIndex()

        # connect to openAI API for embedding
        model_name = 'text-embedding-ada-002'

        embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=config('OPENAI_API_KEY')
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
                embeds = embed.embed_documents(texts)
                index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []

        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))

        success = 'data successfully added: {file}<br><br>'.format(file=filename)

    # no file detected, return the HTML to prompt for a download
    return '''
        <html>
            <body>
                <p>{txt}Select the .txt file you want to sum up:<br>this will take a while, please be patient while it loads</p>
                <form method="post" action="/updateDataset" enctype="multipart/form-data">
                    <p><input type="file" accept=".txt" name="input_file" /></p>
                    <p><input type="submit" value="Process the file" /></p>
                </form><br>
                <form action="/">
                    <input type='submit' value='Home'>
                </form>
            </body>
        </html>
    '''.format(txt=success)

# the home page for the website
@app.route('/')
def start():
    return '''
        <html><body>
            <form action="/question">
                 <input type='submit' value='Ask a Question'>
             </form>

             <form action="/updateDataset">
                 <input type='submit' value='Add Documents to Dataset'>
             </form>

             <form action="/deletePrompt">
                 <input type='submit' value='Clear the Dataset (this will take a long time, be patient)'>
             </form>
        </body></html>
    '''

# prompts the user to ask a question, then sends the user to the answer function
@app.route('/question')
def question():
    # this is pretty much the entire website right here
    HOME_HTML = """
     <html><body>
         <h2>Chatting With ChatGPT</h2>
         <form action="/answer">
             enter your prompt for ChatGPT: <input type='text' name='prompt'><br>
             <input type='submit' value='Continue'>
         </form><br>
         <form action="/">
             <input type='submit' value='Home'>
         </form>
     </body></html>"""
    return HOME_HTML

# sends the prompt from question through the qa chain and displays the answer with sources
@app.route('/answer')
def answer():
    # get the qa chain set in setup()
    global qa

    # retrive the prompt from the url
    inp = request.args.get('prompt', '')

    # will return source documents and answer
    result = qa({"query": inp})
    resp = result["result"]
    source = result["source_documents"]

    # make the source a bit prettier for html
    source = str(source)
    source = source.replace("\\n", "<br>")
    source = source.replace("\\r", "")
    source = source.replace("Document(page_content=\'", "<br><br>NEXT SOURCE<br>")

    return """
     <html><body>
         <h2>Chatting With ChatGPT</h2>
         <form action="/question">
             prompt: {input}<br>
             answer: {response}<br>
             <input type='submit' value='Continue'><br>
             sources: {src}
         </form><br>
         <form action="/">
             <input type='submit' value='Home'>
         </form>
     </body></html>""".format(input=inp, response=resp, src=source)

# display every file in the database and when you click on one, it gets deleted
@app.route('/deletePrompt')
def chooseWhichToDelete():
    # load filenames
    files = getFileNames()

    # generate html string
    buttons = ''
    buttonTemplate = "<button name=\"filename\" type=\"submit\" value=\"{file}\">{file}</button><br>"
    for i in files:
        buttons = buttons + buttonTemplate.format(file=i)

    return """
    <html><body>
        <form action="/deleteAction">
          Choose which file to delete:<br>
          {b}
        </form>
    </body></html>""".format(b=buttons)

@app.route('/deleteAction')
def deleteByMetaData():
    # get filename from html
    filename = request.args.get('filename', '')

    # remove file from json document
    removeFileFromJSON(filename)

    # connect to index
    index = connectToIndex()

    # delete
    index.delete(
        filter={
            "filename": filename
        }
    )

    return showText('success!')

# deletes all vectors from the index by deleting the index
@app.route('/deleteAll')
def deleteAllVectors():
    # since we are resetting the index, we need to reset the qa thing as well

    global qa

    index_name = 'langchain-retrieval-augmentation-stroke'

    pinecone.delete_index(index_name)

    # make new index, now empty
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

    # run setup again so pinecone can connect to the correct index
    qa = getQA()

    return '''
        <html>
            <body>
                <form action="/">
                    <p>delete successful!</p>
                     <input type='submit' value='Home'>
                 </form>
            <body>
        <html>
        '''

# initialize the index and return the qa machine
def getQA():
    # connect to the index
    index = connectToIndex()

    # connect to openAI API for embedding
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=config('OPENAI_API_KEY')
    )

    # create a vectorstore using pinecone
    text_field = "text"

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    # use langchain to make a chain to do all the hard work for us
    # completion llm
    llm = ChatOpenAI(
        openai_api_key=config('OPENAI_API_KEY'),
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    # return the chain used
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def setup():
    # set up proxy server so we can use pinecone through pythonanywhere (the python hosting service I use)
    openapi_config = OpenApiConfiguration.get_default_copy()
    openapi_config.proxy = "http://proxy.server:3128"

    pinecone.init(
        api_key=config('PINECONE_API_KEY'),
        environment=config('PINECONE_ENVIRONMENT'),
        openapi_config=openapi_config
    )

    # connect to and return the index
    return getQA()

# returns the pinecone index we use for the database
def connectToIndex():
    # index name
    index_name = 'langchain-retrieval-augmentation-stroke'

    # if index does not exist, create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

    # connect to the index and return it
    return pinecone.Index(index_name)

# set up the tokenizer
tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

# token length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def showText(text):
    return '''
        <html>
            <body>
                <p>{txt}
            <body>
        <html>
        '''.format(txt=text)

# loads the JSON file with the filenames of all docs currently in the dataset and returns a list of them
def getFileNames():
    # path to json file
    global namesPath

    # load the JSON file and convert to a dictionary
    dict = json.load(open(namesPath))

    # check if there's an item called files (create one if not)
    if 'files' not in dict:
        # make it there!
        dict['files'] = []

    print(dict)

    # return the list
    return dict['files']

# adds a file to the list of json files and returns the name of the file added
def addFileToJSON(filename):
    # path to json file
    global namesPath

    # load the current info in the json file
    files = getFileNames()

    # check for duplicate filename
    if filename in files:
        # keep on trying to add filename(1) or filename(2) until no duplicate is found
        num=1
        while (filename + "({n})".format(n=num)) in files:
            num=num+1

        filename = filename + "({n})".format(n=num)

    # append new filename to list
    files.append(filename)

    # put into dictionary
    dict = {'files': files}

    # send dictionary back to json file
    with open(namesPath, "w") as outfile:
        json.dump(dict, outfile)

    return filename

# removes the filename from the JSON file
def removeFileFromJSON(filename):
    # path to json file
    global namesPath

    # verify that filename is in documents
    files = getFileNames()

    if filename not in files:
        return showText('the file you have requested to delete is not in the dataset')

    # remove the desired file from the list
    files.remove(filename)

    # put into dictionary
    dict = {'files': files}

    # send dictionary back to json file
    with open(namesPath, "w") as outfile:
        json.dump(dict, outfile)


# initialize the qa and set up all the API's that need setting up
qa = setup() # the chain used to send prompts