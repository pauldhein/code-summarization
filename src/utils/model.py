import dynet as dy

import utils.utils as utils


class AttentionEncoderDecoder():
    def __init__(self):
        self.model = dy.Model()

        self.code_data, self.nl_data = utils.load_parallel_corpus("./data/")
        self.code_embed, self.comm_embed = utils.get_pretrained_embeddings("./vector-maker/")
