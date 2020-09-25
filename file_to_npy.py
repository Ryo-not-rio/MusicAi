import numpy as np
import ast


def to_npy(input_file="data.txt", output_file="data.npy"):
    with open(input_file) as f:
        r = f.read()

    data = r.split(";")[:-1]

    data = np.array(list(map(lambda x: ast.literal_eval(x), data)))

    with open(output_file, 'wb') as f:
        np.save(f, data)

