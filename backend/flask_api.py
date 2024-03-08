from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


from flask import Flask, jsonify, request
from flask_cors import CORS

# Create a Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.environ["OPENAI_API_KEY"]

# model_name = "BAAI/bge-large-en-v1.5"
# embeddings_huggingface = HuggingFaceEmbeddings(model_name = model_name)
embeddings_openai = OpenAIEmbeddings()

def get_chunks():
    folder_name = 'Bio12th_NCERT'

    docs = []
    for file in os.listdir(folder_name):
        filename = os.path.join(folder_name, file)
        print("filename --> ",filename)
        if not filename.endswith('.pdf'):
            continue
        with pdfplumber.open(filename) as pdf:
            for indx,page in enumerate(pdf.pages):
                text = page.extract_text()

                splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
                for chunk in splitter.split_text(text):

                    doc =  Document(page_content=chunk, metadata={"source": file, "page_number":indx})
                    docs.append(doc)
    return docs

def create_hugging_embedding(embeddings):
    # model_name = "BAAI/bge-large-en-v1.5"
    # embeddings = HuggingFaceEmbeddings(model_name = model_name)
    db_hugging = FAISS.from_documents(docs, embeddings)
    db_hugging.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)


def create_openai_embedding(embeddings):
    # embeddings = OpenAIEmbeddings()
    db_openai = FAISS.from_documents(docs, embeddings)
    db_openai.save_local("faiss_index_openai")
    new_db_openai = FAISS.load_local("faiss_index_openai", embeddings)


def final_llm(query, db , k=2):

  prompt = '''You are a smart bot which provide answer to queries related to biology form the provided context,
  query and the context from which the answer is to be given are provided below. PLEASE Stick to the context and DO NOT hallucinate.

  query - {}
  context - {}

  answer -
  '''


  docs = db.similarity_search(query, k)
  context = ''


  
  for doc in docs:
    # print("Docs --> ", doc.page_content,"\n")
    context = context + doc.page_content

  client = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY"),
  )

  chat_completion = client.chat.completions.create(
      messages=[
          {
        "role": "user",
        "content": prompt.format(query,context)
      }
          ],
      model="gpt-3.5-turbo",
  )
  return chat_completion

# Define a route for the API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    # Return a JSON response
    return jsonify({'message': 'Hello, World!'})

@app.route('/call/openai', methods=['GET','POST'])
def call_openai():
    
    print("in openai call")
    db = FAISS.load_local("faiss_index_openai", embeddings_openai)

    data = request.get_json()
    query = data["query"]
    print("question --> ", query)

    res = final_llm(query,db,7)
    output = res.choices[0].message.content

    print("final output --> ",output)
    return jsonify({"resp":output})


# @app.route('/call/huggingface', methods=['GET'])
# def call_huggingface():
    
#     db = FAISS.load_local("faiss_index", embeddings_huggingface)

#     data = request.get_json()
#     query = data["query"]

#     res = final_llm(query,db,7)
#     output = res.choices[0].message.content
#     return jsonify({"resp":output})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=4500)

