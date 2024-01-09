import PyPDF2
import json
import os
import aspose.words as aw

from time import time
from tika import parser
from tika import tika

tika.TikaJarPath = "tika/tika-server.jar"

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

            with open("Documents/Parsed/" + filename + ".txt","w",encoding="utf-8") as f_out:
                f_out.write(text)

    time_2 = time()
    print(f"Parsing Time: {round(time_2-time_1, 3)} sec.")
    input("Files parsed. They can now be deleted. Press enter to continue...")


upload_files("","main")