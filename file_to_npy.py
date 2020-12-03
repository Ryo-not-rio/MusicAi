"""
Converting all data text files into numpy files.
"""

import numpy as np
import ast
import os


def to_npy(input_file="data.txt", output_file="data.npy"):
    with open(input_file) as f:
        r = f.read()

    data = r.split(";")[:-1]

    data = np.array(list(map(lambda x: ast.literal_eval(x), data)))

    with open(output_file, 'wb') as f:
        np.save(f, data)


def convert_all(in_directory="data"):
    directory = os.fsencode(in_directory)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        to_npy(input_file=os.path.join(in_directory, file_name), output_file=os.path.join(in_directory, file_name[:-3]+"npy"))

# if __name__ == "__main__":
    # convert_all()
    # lengths = []
    # directory = os.fsencode("data")
    # for file in os.listdir(directory):
    #     file_name = os.fsdecode(file)
    #     if file_name[-3:] == "npy":
    #         with(open(os.path.join("data", file_name), "rb")) as f:
    #             print(np.load(f).size)