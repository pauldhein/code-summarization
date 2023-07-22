import pickle
import os

import numpy as np


def build_embedding_array(embeddings, int2emb):
    emb_arr = list()
    for i in range(len(embeddings.keys())):
        symbol = int2emb[i]
        embedding = embeddings[symbol]
        emb_arr.append(embedding)

    return np.array(emb_arr)


def load_parallel_corpus(data_path):
    train_code, train_comm = list(), list()
    val_code, val_comm = list(), list()
    test_code, test_comm = list(), list()

    with open(os.path.join(data_path, "train.code"), "r") as infile:
        for line in infile.readlines():
            train_code.append(line.strip().split(" "))

    with open(os.path.join(data_path, "train.nl"), "r") as infile:
        for line in infile.readlines():
            train_comm.append(line.strip().split(" "))

    with open(os.path.join(data_path, "dev.code"), "r") as infile:
        for line in infile.readlines():
            val_code.append(line.strip().split(" "))

    with open(os.path.join(data_path, "dev.nl"), "r") as infile:
        for line in infile.readlines():
            val_comm.append(line.strip().split(" "))

    with open(os.path.join(data_path, "test.code"), "r") as infile:
        for line in infile.readlines():
            test_code.append(line.strip().split(" "))

    with open(os.path.join(data_path, "test.nl"), "r") as infile:
        for line in infile.readlines():
            test_comm.append(line.strip().split(" "))

    return {"train": train_code, "val": val_code, "test": test_code}, \
        {"train": train_comm, "val": val_comm, "test": test_comm}


def get_vocabs(utils_path):
    pkl_path = os.path.join(utils_path, "code-comment-data.pkl")
    code_comm_data = pickle.load(open(pkl_path, "rb"))
    (code, comm) = map(list, zip(*list(code_comm_data.values())))
    code_syms = [sym for code_set in code for sym in code_set]
    comm_syms = [sym for comm_set in comm for sym in comm_set]
    return list(set(code_syms)), list(set(comm_syms))


def get_pretrained_embeddings(vector_path):
    code_path = os.path.join(vector_path, "code-vectors.txt")
    comm_path = os.path.join(vector_path, "comm-vectors.txt")

    def get_embeddings(emb_path):
        embeddings = dict()
        with open(emb_path, "r") as infile:
            for idx, line in enumerate(infile.readlines()):
                if idx == 0:
                    continue
                cells = line.strip().split(" ")
                symbol = cells[0]
                embedding = [float(cell) for cell in cells[1:]]
                embeddings[symbol] = embedding
        return embeddings

    return get_embeddings(code_path), get_embeddings(comm_path)
