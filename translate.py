from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from evaluate import evaluate_model
import math
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

src, trg = read_nmt_data(
    src=config['data']['src'],
    config=config,
    trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    trg=config['data']['test_trg']
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()


model = Seq2SeqAttention(
    src_emb_dim=config['model']['dim_word_src'],
    trg_emb_dim=config['model']['dim_word_trg'],
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_hidden_dim=config['model']['dim'],
    trg_hidden_dim=config['model']['dim'],
    ctx_hidden_dim=config['model']['dim'],
    attention_mode='dot',
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    pad_token_src=src['word2id']['<pad>'],
    pad_token_trg=trg['word2id']['<pad>'],
    nlayers=config['model']['n_layers_src'],
    nlayers_trg=config['model']['n_layers_trg'],
    dropout=0.,
).cuda()

model.load_state_dict(torch.load(open(sys.argv[1])))

preds = evaluate_model(
    model, src, src_test, trg,
    trg_test, config, verbose=False,
    metric='bleu',raw=True
)

print preds