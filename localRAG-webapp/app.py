from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
from torch import cuda, bfloat16
import torch
import transformers
from langchain.schema.runnable import RunnablePassthrough
from transformers import AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import GPT2Model, GPT2Tokenizer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import  ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from transformers import AutoModel
from transformers import GPT2TokenizerFast
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForQuestionAnswering
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain import OpenAI
import shutil
import os
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from time import time
from tika import parser
from tika import tika
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

import openai

tika.TikaJarPath = "tika/"

app = Flask(__name__)
app.config["TIMEOUT"] = 5000
app.config['SECRET_KEY'] = 'your_secret_key'
login_manager = LoginManager(app)
login_manager.login_view = 'login'

#GLOBAL MODULES
vectordb = None
tokenizer = None
qa_pipeline = None
retriever = None
llm = None

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    return users.get(int(user_id))

def load_users():
    with open("users.json", "r") as f_in:
        users_data = json.load(f_in)

    users = {user['id']: User(**user) for user in users_data}
    return users

@app.route('/initialise', methods=["GET","POST"])
def initialise(): #Initialising everything
    # Assuming you have a global variable to store the vector database instance
    # You might want to use a database connection pool for a production application
    # global vectordb
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {"device": "cuda"}
    # embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    # # Access the current logged-in user's username
    # current_username = current_user.username if current_user.is_authenticated else "guest"
    # # Use the username to initialize the vector database
    # vectordb = Chroma(persist_directory=f"chroma_db_{current_username}", embedding_function=embeddings)
    print("initialise triggered")
    try:
        #INITIALISING VECTORDB
        user = current_user.username if current_user.is_authenticated else None
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        global vectordb
        vectordb = Chroma(persist_directory="chroma_db_" + user, embedding_function=embeddings)

        print("vectordb initilized")
        #INITIALISING OPENAI chat completion model
        # api_key = "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"
        # global llm
        # llm = llm_initialize(api_key, llm_model)
        # print(llm)

        global tokenizer
        # tokenizer = AutoTokenizer.from_pretrained('MBZUAI/LaMini-Neo-125M')
        tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
        print("tokenizer initialized")

        model_id = "Intel/dynamic_tinybert"
        # global qa_pipeline
        # qa_pipeline = pipeline('text-generation', model=llm_model, tokenizer=tokenizer)
        # print("qa pipeline initialized")
        # model_config = transformers.AutoConfig.from_pretrained(model_id)
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     trust_remote_code=True,
        #     config= model_config,
        #     device_map="auto"
        # )

        # model_name = "Intel/dynamic_tinybert"
        # model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

        # query_pipeline = transformers.pipeline(
        #     "text-generation",
        #     model = model,
        #     tokenizer = tokenizer,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     max_new_tokens=4096,
        #     batch_size=16
        # )

        # global llm
        # # llm = HuggingFacePipeline(pipeline=query_pipeline)
        # llm = HuggingFacePipeline(
        #     pipeline=question_answerer,
        #     model_kwargs={"temperature": 0.7, "max_length": 512},
        # )
        # print("llm initialsied")

        #INITIALISING LLM MODEL

        global llm
        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # llm = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        # print(llm)
        # tokenizer = AutoTokenizer.from_pretrained("Arc53/docsgpt-7b-mistral")
        # llm = AutoModelForCausalLM.from_pretrained("Arc53/docsgpt-7b-mistral")
        # llm = BertModel.from_pretrained('bert-base-uncased')
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # llm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        # print("llm ready")

        #INITIALISING RETRIEVER
        global retriever
        retriever = vectordb.as_retriever()

        return jsonify({'success': True, 'message': 'Initialization successful'})
    except Exception as e:
        return jsonify({'success': False, 'error_message': str(e)})

@app.route('/')
@login_required
def home():
    return render_template("index.html")

def llm_initialize(api_key, model):
    openai.api_key = api_key
    return {"model": model}

@app.route('/check_files', methods=["GET","POST"])
def check_files():
    try:
        print("check_files triggered")
        uploaded_files_path = "uploaded_files"
        processed_files_path = "vectorized_documents.json"

        # Get the list of files in the uploaded_files folder
        uploaded_files = [file for file in os.listdir(uploaded_files_path) if os.path.isfile(os.path.join(uploaded_files_path, file))]

        user = current_user.username if current_user.is_authenticated else None
        with open(processed_files_path, 'r') as f:
            processed_files_data = json.load(f)
            processed_files = processed_files_data[user]

        print(uploaded_files)
        print(processed_files)
        return jsonify({
            'uploadedFiles': uploaded_files,
            'processedFiles': processed_files
        })

    except Exception as e:
        return "hi"

@app.route('/query_chain', methods=["GET", "POST"])
def query_chain():
    try:
        print("Query chain triggered")

        # retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        retriever = vectordb.as_retriever()
        print(retriever)
        query = request.json.get("query")

        template = """Given the question: {question}
                        Answer based on the given context. If no relevant information in the context, state so: {context}
                        """

        # print(llm)
        # llm = OpenAI(temperature=0.0, openai_api_key="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", model_name="gpt-3.5-turbo-16k")
        llm = OpenAI(temperature=0.0, openai_api_key="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", model_name="gpt-3.5-turbo-16k")
        print(llm)
        # llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        # llm = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            verbose=True,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
            }
        )

        print("test")
        sources = {}
        source_index = 1
        source = ""
        result = "No relevant context found"
        docs = retriever.get_relevant_documents(query)
        print(docs)
        if len(docs) > 0:
            print(f"\nRetrieved documents: {len(docs)}")
            for doc in docs:
                print(docs)
                doc_details = doc.to_json()['kwargs']
                source = "Source filename: " + str(doc_details['metadata']['source'] + "\n\n\n")
                content = str(doc_details['page_content'])
                source = source + "\n\n" + str(content)

                sources[source_index] = source
                # sources_content += doc_details['page_content']
                source_index += 1

        result = qa_chain.run(query)
        print(result)

        # sources_content = ""

        # FILTERING DOCS WITH DISTANCE < 1.2 -> HIGHER SIMILARITY
        # docs = [(content, score) for content, score in docs if score <= 1.7]

        # IF THERE ARE RELEVANT VECTORS, THEN FEED AND QUERY LLM FOR ANSWER
        # if len(docs) > 0:
        #     print(f"\nRetrieved documents: {len(docs)}")
        #     for doc in docs:
        #         doc = doc[0]
        #         print(docs)
        #         doc_details = doc.to_json()['kwargs']
        #         # # print("\nSource: ", doc_details['metadata']['source'])
        #         # # print("Text: ", doc_details['page_content'], "\n")
        #         # sources = sources + doc_details['metadata']['source']
        #         # sources = sources + doc_details['page_content']
        #         # sources = sources + "\n\n\n"
        #         source = source + doc_details['metadata']['source']
        #         source = source + doc_details['page_content']
        #         sources[source_index] = source
        #         # sources_content += doc_details['page_content']
        #         source_index += 1
        #     # time_1 = time()
        #     # result = qa_chain.run(query)
        #
        # else:
        #     print("No relevant vectors found")
        #
        # print(sources)

        response_data = {
            "result": result,
            "sources": sources
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)})

    # # Run the query loop.
    # while query_done == False:
    #     os.system("cls||clear")
    #     query = input("Please enter your query: ")
    #     time_1 = time()
    #     result = qa.run(query)
    #     time_2 = time()
    #
    #     os.system("cls||clear")
    #     print(f"Inference time: {round(time_2 - time_1, 3)} sec.")
    #     print("\nQuery: ", query)
    #     print("\nAnswer: ", result)

        # source = input("\nWould you like to see the sources the answer is taken from? (y/n): ")
        #
        # match source.lower():
        #     case "y":
        #         docs = vectordb.similarity_search(query)
        #         print(f"\nRetrieved documents: {len(docs)}")
        #
        # for doc in docs:
        #     doc_details = doc.to_json()['kwargs']
        #     print("\nSource: ", doc_details['metadata']['source'])
        #     print("Text: ", doc_details['page_content'], "\n")
        #
        #
        # cont = input("\nDo you want to ask another question? (y/n): ")
        #
        # match cont.lower():
        #     case "n":
        #         query_done = True
        #         print("Returning to main menu...")

@app.route('/login_action', methods=["GET", "POST"])
def login_action():
    print("login action triggered")
    username = request.json.get('username')
    password = request.json.get('password')
    print(username)
    print(password)
    users = load_users()

    user = next((user for user in users.values() if user.username == username and user.password == password), None)
    if user:
        print("user exists")
        login_user(user)
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        print("login failed")
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/login', methods=["GET", "POST"])
def login():
    return  render_template("login.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    # Clear the vector database instance upon logout
    global vectordb
    vectordb = None
    return redirect(url_for('login'))

@app.route('/chromadb')
def chroma():
    ids_to_delete = []
    coll = vectordb.get()
    for idx in range(len(coll['ids'])):
        id = coll['ids'][idx]
        metadata = coll['metadatas'][idx]
        print(metadata)
        ids_to_delete.append(id)

    # vectordb._collection.delete(ids_to_delete)
    return 'hi'

@app.route('/delete_uploaded_document', methods=["POST"])
def delete_uploaded_document():
    print("delete uploaded document triggered")
    try:
        # Parse the JSON data from the request
        data = request.get_json()

        if 'fileName' in data:
            # Obtain the filename
            filename = data['fileName']
            basename, file_extension = os.path.splitext(filename)

            # Deleting from uploaded files folder
            upload_folder_path = "uploaded_files"
            upload_file_path = upload_folder_path + "/" + filename
            print(upload_file_path)
            if os.path.exists(upload_file_path):
                print("exists")
                os.remove(upload_file_path)
                print(f'File {filename} deleted from uploaded_files.')

            # Deleting from parsed files folder
            parsed_folder_path = "parsed_files"
            parsed_file_path = parsed_folder_path + "/" + basename + ".txt"
            print(parsed_file_path)
            print(os.path.exists(parsed_file_path))
            if os.path.exists(parsed_file_path):
                print("exists")
                os.remove(parsed_file_path)
                print(f'File {filename} deleted from parsed_files.')

        return jsonify({'message': f'Document {filename} deleted successfully'})

    except Exception as e:
        # Handle exceptions if any occur during the process
        return jsonify({'error': str(e)}), 500

@app.route('/delete_document', methods=["POST"])
@login_required
def delete_document():
    print("delete document triggered")
    try:
        # Parse the JSON data from the request
        data = request.get_json()


        # Check if the 'fileName' key exists in the data
        if 'fileName' in data:
            # Obtain the filename
            filename = data['fileName']
            basename, file_extension = os.path.splitext(filename)
            print(filename)
            # # Check if the file exists in the 'uploaded_files' directory
            # uploaded_file_path = os.path.join(app.config['uploaded_files'], filename)
            # if os.path.exists(uploaded_file_path):
            #     print("exists")
            #     # Delete the file from 'uploaded_files'
            #     os.remove(uploaded_file_path)
            #     print(f'File {filename} deleted from uploaded_files.')

            #Deleting from uploaded files folder
            upload_folder_path = "uploaded_files"
            upload_file_path = upload_folder_path + "/" + filename
            print(upload_file_path)
            if os.path.exists(upload_file_path):
                print("exists")
                os.remove(upload_file_path)
                print(f'File {filename} deleted from uploaded_files.')

            #Deleting from parsed files folder
            parsed_folder_path = "parsed_files"
            parsed_file_path = parsed_folder_path + "/" + basename + ".txt"
            print(parsed_file_path)
            print(os.path.exists(parsed_file_path))
            if os.path.exists(parsed_file_path):
                print("exists")
                os.remove(parsed_file_path)
                print(f'File {filename} deleted from parsed_files.')

            # # Check if the file exists in the 'parsed_files' directory
            # parsed_file_path = os.path.join(app.config['parsed_files'], filename)
            # if os.path.exists(parsed_file_path):
            #     # Delete the file from 'parsed_files'
            #     os.remove(parsed_file_path)
            #     print(f'File {filename} deleted from parsed_files.')

            #Run through chroma to delete chunks of document
            ids_to_delete = []
            coll = vectordb.get()
            print(coll)
            for idx in range(len(coll['ids'])):
                id = coll['ids'][idx]
                metadata = coll['metadatas'][idx]
                source = metadata["source"]
                source = source.replace("parsed_files\\", "")
                print(source)
                source = source.split(".")[0]
                filename = filename.split(".")[0]

                if str(filename) == str(source):
                    print("filename in source. deleting from vectordb")
                    ids_to_delete.append(id)

            vectordb._collection.delete(ids_to_delete)

            #REMOVE FROM VECTORIZED_DOCUMENT.JSON
            vectorized_documents_file_path = 'vectorized_documents.json'
            user = current_user.username if current_user.is_authenticated else None

            with open(vectorized_documents_file_path, 'r') as file:
                data = json.load(file)

            vectorized_docs = data[user]
            if basename in vectorized_docs:
                vectorized_docs.remove(basename)
            data[user] = vectorized_docs

            with open(vectorized_documents_file_path, 'w') as file:
                json.dump(data, file, indent=2)

            print("Document removed from vectorized_document json records")
            print("Document vectors deleted from Chromadb")

            return jsonify({'message': f'Document {filename} deleted successfully'})
        else:
            # If 'fileName' is not provided in the request data, return an error message
            return jsonify({'error': 'Filename not provided in the request data'}), 400

    except Exception as e:
        # Handle exceptions if any occur during the process
        return jsonify({'error': str(e)}), 500

@app.route('/vectorise')
@login_required
def vectorise():
    try:
        print("vectorising triggered")
        # Initialize Embeddings
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_kwargs = {"device": "cuda"}
        # embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        # Assuming you have a User model with a 'username' attribute
        user = current_user.username if current_user.is_authenticated else None

        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

        vectordb = Chroma(persist_directory="chroma_db_" + user, embedding_function=embeddings)

        # Load vectorized_documents
        with open("vectorized_documents.json", "r") as f_in:
            vectorized_documents = json.load(f_in)

        directory = "parsed_files"
        # Iterate through every text to vectorize.
        time_1 = time()
        for filename in os.listdir(directory):
            print(filename)
            [name, extension] = filename.split(".")

            if (name) in vectorized_documents[user]:
                print("File already vectorized, skipping...")
            else:
                # Add to the vectorized documents database for checking.
                vectorized_documents[user].append(name)

                # Ingest the text.
                loader = TextLoader(
                    os.path.join(directory, filename),
                    encoding="utf-8"
                )
                documents = loader.load()

                # Split the text into chunks.
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
                all_splits = text_splitter.split_documents(documents)

                # Vectorize the text.
                vectordb.add_documents(all_splits)
                torch.cuda.empty_cache()

        # Update vectorized_documents
        with open("vectorized_documents.json", "w") as f_out:
            json.dump(vectorized_documents, f_out)

        time_2 = time()
        print(f"Vectorizing Time: {round(time_2 - time_1, 3)} sec.")
        print("Files vectorized. They can now be deleted.")

        return jsonify({'message': 'Files vectorised successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/reset', methods=["GET", "POST"])
def reset():
    print("reset triggered")
    try:
        user = current_user.username if current_user.is_authenticated else None
        # Delete chroma_db_ folder
        chroma_db_folder = f"chroma_db_{user}"

        if os.path.exists(chroma_db_folder):
            print("exist")
            shutil.rmtree(chroma_db_folder)
            print(f"{chroma_db_folder} deleted successfully")

        # Delete files in uploaded_files folder
        uploaded_files_folder = "uploaded_files"
        for file_name in os.listdir(uploaded_files_folder):
            file_path = os.path.join(uploaded_files_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")

        # Delete files in parsed_files folder
        parsed_files_folder = "parsed_files"
        for file_name in os.listdir(parsed_files_folder):
            file_path = os.path.join(parsed_files_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")

        print("test")
            # Reset vectorized_documents.json data
        with open("vectorized_documents.json", "w") as f_out:
            json.dump({"default": []}, f_out)
        print("test2")
        return redirect(url_for("logout"))
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if the 'file' key is in the request.files dictionary
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        uploaded_files = request.files.getlist('file')

        # Create the 'uploaded_files' folder if it doesn't exist
        upload_folder = 'uploaded_files'
        os.makedirs(upload_folder, exist_ok=True)

        # Save each uploaded file to the 'uploaded_files' folder
        for file in uploaded_files:
            file.save(os.path.join(upload_folder, file.filename))

        print("Files uploaded successfully")

        # Create the 'parsed_files' folder if it doesn't exist
        parsed_folder = 'parsed_files'
        os.makedirs(parsed_folder, exist_ok=True)

        time_1 = time()

        directory = "uploaded_files"
        for filename in os.listdir(directory):
            [name, extension] = filename.split(".")
            print(filename)
            if (name + ".txt") in os.listdir("parsed_files"):
                print("File already parsed, skipping...")
            else:
                parsed = parser.from_file(os.path.join(directory, filename))
                text = parsed['content']

                with open("parsed_files/" + name + ".txt", "w", encoding="utf-8") as f_out:
                    f_out.write(text)

        time_2 = time()
        print(f"Parsing Time: {round(time_2 - time_1, 3)} sec.")
        return jsonify({'message': 'Files parsed and saved successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True, port=8000)