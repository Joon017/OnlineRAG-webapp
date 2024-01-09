## https://www.kaggle.com/code/ferhat00/rag-on-financial-10-q-statements-using-llama2

import json
import os

from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from time import time
from tika import parser
from tika import tika

tika.TikaJarPath = "tika/"

# Function Space

exit_now = False

def llm_initialize():

    # Define the model, device, and configuration.
    model_id = 'daryl149/llama-2-7b-chat-hf'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # Prepare the model and Tokenizer.
    time_1 = time()
    model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    time_2 = time()
    print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")

    # Define the Pipeline.
    time_1 = time()
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=4096,
        batch_size=16
    )
    time_2 = time()
    print(f"Prepare pipeline: {round(time_2-time_1, 3)} sec.")

    llm = HuggingFacePipeline(pipeline=query_pipeline)

    return llm


def upload_files(vectordb, user):

    # Load vectorized_documents
    with open("vectorized_documents.json","r") as f_in:
        vectorized_documents = json.load(f_in)

    os.system("cls||clear")
    print("Please place all the documents you wish to transcribe into the \"Documents/Unparsed\" folder...")
    input("Press enter to continue...")

    directory = "Documents/Unparsed/"

    # Iterate through every document to convert to text.
    time_1 = time()

    for filename in os.listdir(directory):
        [name, extension] = filename.split(".")
        print(filename)
        if (name + ".txt") in os.listdir("Documents/Parsed/") or (name) in vectorized_documents[user]:
            print("File already parsed, skipping...")
        else:
            parsed = parser.from_file(os.path.join(directory, filename))
            text = parsed['content']

            with open("Documents/Parsed/" + name + ".txt","w",encoding="utf-8") as f_out:
                f_out.write(text)

    time_2 = time()
    print(f"Parsing Time: {round(time_2-time_1, 3)} sec.")
    input("Files parsed. They can now be deleted. Press enter to continue...")
    os.system("cls||clear")

    directory = "Documents/Parsed/"

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
    with open("vectorized_documents.json","w") as f_out:
        json.dump(vectorized_documents,f_out)

    time_2 = time()
    print(f"Vectorizing Time: {round(time_2-time_1, 3)} sec.")
    input("Files vectorized. They can now be deleted. Press enter to continue...")
    os.system("cls||clear")


def query_chain(llm):
    
    # Initialize QA Chain
    retriever = vectordb.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )

    query_done = False

    # Run the query loop.
    while query_done == False:
        os.system("cls||clear")
        query = input("Please enter your query: ")
        time_1 = time()
        result = qa.run(query)
        time_2 = time()

        os.system("cls||clear")
        print(f"Inference time: {round(time_2-time_1, 3)} sec.")
        print("\nQuery: ", query)
        print("\nAnswer: ", result)

        source = input("\nWould you like to see the sources the answer is taken from? (y/n): ")

        match source.lower():
            case "y":
                docs = vectordb.similarity_search(query)
                print(f"\nRetrieved documents: {len(docs)}")

                for doc in docs:
                    doc_details = doc.to_json()['kwargs']
                    print("\nSource: ", doc_details['metadata']['source'])
                    print("Text: ", doc_details['page_content'], "\n")

        cont = input("\nDo you want to ask another question? (y/n): ")

        match cont.lower():
            case "n":
                query_done = True
                print("Returning to main menu...")
        
        torch.cuda.empty_cache()


def list_documents(user):
    # Load vectorized_documents
    with open("vectorized_documents.json","r") as f_in:
        vectorized_documents = json.load(f_in)

    # Print them out.
    os.system("cls||clear")
    print("Documents Stored in Vector Database.\n")
    for doc in vectorized_documents[user]:
        print(doc)

    input("\nPress enter to continue...")


if __name__ == "__main__":
    
    # User Initialization
    with open("users.json","r") as f_in:
        users = json.load(f_in)
    
    user = ""

    while user not in users:
        os.system("cls||clear")
        user = input("Please enter your username: ").lower()

        if user not in users:
            input("User not found. Press enter to try again.")

    print("Welcome, " + user + "!")

    # Initialize Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # Initialize VectorDB
    print("Initializing chroma_db_" + user)
    vectordb = Chroma(persist_directory="chroma_db_" + user, embedding_function=embeddings)

    # Initialize LLM
    llm = llm_initialize()

    while exit_now != True:
        os.system("cls||clear")
        print("RAG Local e Youkosou!")
        print("What would you like to do?")
        print("> Upload")
        print("> Query")
        print("> List")
        print("> Exit")
        choice = input("Please enter your choice: ")

        match choice.lower():
            case "exit":
                print("Exiting...")
                exit_now = True
                os.system("cls||clear")
            case "upload":
                upload_files(vectordb, user)
            case "query":
                query_chain(llm)
            case "list":
                list_documents(user)
