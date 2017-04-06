from data_utils import read_test_data, get_minibatch, read_config

import os
import sys

import torch

config_file_path = args.config
config = read_config(config_file_path)


def test_model(model, data_file, nwords, max_length, batch_size):
    
    predictions = []
    src, trg = read_test_data(src = data_file, n_words = nwords)

    input_lines_src, _, lens_src, mask_src = get_minibatch(
        src['data'], src['word2id'], j,
        batch_size, max_length, add_start=True, add_end=True
    )

    input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
        trg['data'], trg['word2id'], j,
        batch_size, max_length, add_start=True, add_end=True
    )

    decoder_logit = model(input_lines_src, input_lines_trg)

    word_probs = model.decode(
        decoder_logit
        ).data.cpu().numpy().argmax(axis=-1)

    output_lines_trg = output_lines_trg.data.cpu().numpy()
    for sentence_pred, sentence_real in zip(word_probs[:5], output_lines_trg[:5]):
        sentence_pred = [trg['id2word'][x] for x in sentence_pred]
        sentence_real = [trg['id2word'][x] for x in sentence_real]

        if '</s>' in sentence_real:
            index = sentence_real.index('</s>')
            sentence_real = sentence_real[:index]
            sentence_pred = sentence_pred[:index]

        predictions.append(' '.join(sentence_pred))
    return predictions


if __name__ == "__main__":
    model = torch.load(sys.argv[1])
    print(model, sys.argv[2], 30000,50, 32)
