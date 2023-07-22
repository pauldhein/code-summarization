import dynet as dy
import random

from tqdm import tqdm

import utils.utils as utils


def main():
    code_data, nl_data = utils.load_parallel_corpus("./data/")
    code_embed, comm_embed = utils.get_pretrained_embeddings("./vector-maker/")
    # code_embed, comm_embed = utils.get_vocabs("./utils/")
    code2int = {code: i for i, code in enumerate(code_embed)}
    lang2int = {lang: i for i, lang in enumerate(comm_embed)}
    # code2int = {code: i for i, code in enumerate(code_embed.keys())}
    # lang2int = {lang: i for i, lang in enumerate(comm_embed.keys())}
    int2code = {i: word for word, i in code2int.items()}
    int2lang = {i: word for word, i in lang2int.items()}

    code_emb_arr = utils.build_embedding_array(code_embed, int2code)
    lang_emb_arr = utils.build_embedding_array(comm_embed, int2lang)

    CODE_VOC_SIZE = len(code2int)
    LANG_VOC_SIZE = len(lang2int)
    CODE_EMB_DIM = 50
    LANG_EMB_DIM = 50

    FWD_LSTM_NUM_LAYERS = 2
    BWD_LSTM_NUM_LAYERS = 2
    DEC_LSTM_NUM_LAYERS = 2
    STATE_SIZE = 250
    ATTENTION_SIZE = 250

    model = dy.Model()

    code_emb = model.add_lookup_parameters((CODE_VOC_SIZE, CODE_EMB_DIM))
    code_emb.init_from_array(code_emb_arr)

    enc_fwd_lstm = dy.LSTMBuilder(FWD_LSTM_NUM_LAYERS, CODE_EMB_DIM, STATE_SIZE, model)
    enc_bwd_lstm = dy.LSTMBuilder(BWD_LSTM_NUM_LAYERS, CODE_EMB_DIM, STATE_SIZE, model)
    dec_lstm = dy.LSTMBuilder(DEC_LSTM_NUM_LAYERS, STATE_SIZE*2+LANG_EMB_DIM, STATE_SIZE, model)

    attention_w1 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE*2))
    attention_w2 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE*BWD_LSTM_NUM_LAYERS*2))
    attention_v = model.add_parameters((1, ATTENTION_SIZE))
    decoder_w = model.add_parameters((LANG_VOC_SIZE, STATE_SIZE))
    decoder_b = model.add_parameters((LANG_VOC_SIZE))

    lang_emb = model.add_lookup_parameters((LANG_VOC_SIZE, LANG_EMB_DIM))
    lang_emb.init_from_array(lang_emb_arr)

    # ==========================================================================
    # PERFORM TRAINING
    # ==========================================================================
    NUM_EPOCHS = 100

    trainer = dy.AdamTrainer(model)
    training_data = list(zip(code_data["train"], nl_data["train"]))
    for e in range(NUM_EPOCHS):
        random.shuffle(training_data)
        for i, (in_sentence, out_sentence) in enumerate(tqdm(training_data, desc="epoch #{}".format(e+1))):
            dy.renew_cg()

            # Embed the input and output sequences
            emb_input = embed_sentence(in_sentence, code2int, code_emb)

            # Run the encoder to produce and encoding
            encoding = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, emb_input)

            # Run the decoder to produce the loss value
            W = dy.parameter(decoder_w)
            b = dy.parameter(decoder_b)
            W1 = dy.parameter(attention_w1)
            input_mat = dy.concatenate_cols(encoding)

            state = dec_lstm.initial_state()
            embed_prev_output = lang_emb[lang2int[out_sentence[0]]]
            full_input = dy.concatenate([dy.vecInput(STATE_SIZE*2), embed_prev_output])
            state = state.add_input(full_input)

            loss = []
            W1dt = None

            int_out_sentence = [lang2int[s] for s in out_sentence]
            for word in int_out_sentence:
                if W1dt is None:
                    W1dt = W1 * input_mat

                # Attention layer
                W2 = dy.parameter(attention_w2)
                v = dy.parameter(attention_v)
                W2dt = W2 * dy.concatenate(list(state.s()))
                unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(W1dt, W2dt)))
                context = input_mat * dy.softmax(unnormalized)

                vector = dy.concatenate([context, embed_prev_output])
                state = state.add_input(vector)
                probs = dy.softmax(W * state.output() + b)
                loss.append(-dy.log(dy.pick(probs, word)))

                embed_prev_output = lang_emb[word]

            loss = dy.esum(loss)
            loss_val = loss.value()
            loss.backward()
            trainer.update()

        print("Epoch #{}: Loss is {}".format(e+1, loss_val))
    # ==========================================================================
    model.save("enc-dec.model")

    # ==========================================================================
    # PERFORM VALIDATION
    # ==========================================================================
    translations = list()
    for (code_input, lang_input) in zip(code_data["val"], nl_data["val"]):
        dy.renew_cg()

        # Embed the input and output sequences
        emb_input = embed_sentence(code_input, code2int, code_emb)

        # Run the encoder to produce and encoding
        encoding = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, emb_input)

        W = dy.parameter(decoder_w)
        b = dy.parameter(decoder_b)
        W1 = dy.parameter(attention_w1)
        input_mat = dy.concatenate_cols(encoding)

        state = dec_lstm.initial_state()
        embed_prev_output = lang_emb[lang2int[lang_input[1]]]
        full_input = dy.concatenate([dy.vecInput(STATE_SIZE*2), embed_prev_output])
        state = state.add_input(full_input)

        translation = []
        count_EOS = 0
        W1dt = None
        for i in range(len(lang_input)*2):
            if count_EOS == 2:
                break

            if W1dt is None:
                W1dt = W1 * input_mat

            W2 = dy.parameter(attention_w2)
            v = dy.parameter(attention_v)
            W2dt = W2 * dy.concatenate(list(state.s()))
            unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(W1dt, W2dt)))
            context = input_mat * dy.softmax(unnormalized)

            vector = dy.concatenate([context, embed_prev_output])
            state = state.add_input(vector)
            probs = dy.softmax(W * state.output() + b).vec_value()
            next_word = probs.index(max(probs))
            embed_prev_output = lang_emb[next_word]
            chosen_word = int2lang[next_word]

            if chosen_word == "<EoL>":
                count_EOS += 1
                continue

            # if i == 0:
            #     continue

            translation.append(chosen_word)

        translations.append(translation)
    # ==========================================================================

    # ==========================================================================
    # OUTPUT TO TEXT FILE
    # ==========================================================================
    with open("python_trans_test.txt", "w+") as outfile:
        for tran in translations:
            sentence = " ".join(tran)
            outfile.write("{}\n\n".format(sentence))


def embed_sentence(sentence, tok2int, embedding):
    int_sentence = [tok2int[s] for s in sentence]
    emb_sentence = [embedding[i] for i in int_sentence]
    return emb_sentence


def encode_sentence(enc_fwd_rnn, enc_bwd_rnn, emb_sent):
    f_init = enc_fwd_rnn.initial_state()
    b_init = enc_bwd_rnn.initial_state()

    fwd_vecs = f_init.transduce(emb_sent)
    bwd_vecs = b_init.transduce(reversed(emb_sent))

    return [dy.concatenate([f, b]) for f, b in zip(fwd_vecs, reversed(bwd_vecs))]


if __name__ == '__main__':
    main()
