import matplotlib.pyplot as plt
import os

from time import time
from tika import parser
from tika import tika


def map_txt():
    directory = "Documents/Unparsed/"
    breaks = [19, 45, 67, 84, 98]
    x = [10,20,30,40,50]
    y = []

    for i in breaks:

        # Iterate through every document to convert to text.
        time_1 = time()

        for filename in os.listdir(directory)[:i]:
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
        time = time_2 - time_1
        print(f"Parsing Time: {round(time, 3)} sec.")

        y.append(time)

    plt.plot(x,y)

    plt.xlabel('File Size (mb)')
    plt.xlabel('Time (s)')

    plt.title('Speed of File to TXT (HTML)')

    plt.show()
