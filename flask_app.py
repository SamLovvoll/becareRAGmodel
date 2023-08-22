from flask import Flask, request
from backend import backend
from decouple import config
app = Flask(__name__)

bend = backend("/home/Slovvoll/mysite/filenames.json", config('PINECONE_API_KEY'), config('OPENAI_API_KEY'))

# prompts the user to upload a file OR if file is detected, uploads it to the pinecone database
@app.route("/updateDataset", methods=["GET", "POST"])
def download():
    # append this to the normal site text to indicate data was uploaded
    success = ''

    # if file is detected: chunk & upload data to pinecone database
    if request.method == "POST":
        # get the uploaded text file
        input_file = request.files["input_file"]

        # upload the file using the backend
        if bend.uploadFile(input_file):
            success = 'data successfully added<br><br>'
        else:
            success = 'data is not in txt format<br><br>'

    # return the HTML to prompt for a download
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
                 <input type='submit' value='Delete a Document from the Dataset'>
             </form>

             <form action="/deleteAll">
                 <input type='submit' value='Delete All Docs (will take a long time, be patient)'>
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
    # retrive the prompt from the url
    inp = request.args.get('prompt', '')

    # will return source documents and answer
    result = bend.query(inp)
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
    files = bend.getDocs()

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

    # use the backend to delete the document
    bend.deleteDoc(filename)

    return """
    <html><body>
        <form action="/deletePrompt">
            <input type='submit' value='Delete Another Document from the Dataset'>
        </form>
        <form action="/">
            <input type='submit' value='Home'>
        </form>
    </body></html>"""

# deletes all vectors from the index by deleting the index
@app.route('/deleteAll')
def deleteAllVectors():
    bend.deleteAllDocs()
    return """
    <html><body>
        <form action="/">
            all documents deleted
            <input type='submit' value='Home'>
        </form>
    </body></html>"""

def showText(text):
    return '''
        <html>
            <body>
                <p>{txt}
            <body>
        <html>
        '''.format(txt=text)