import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import gfile

import data_utils
from seq2seq_model import Seq2SeqModel

_buckets = [(7, 5), (15, 10), (25, 15), (50, 30)]   # default
data_dir = "../data/data0"

def create_model(session, srce_vocab_size, trgt_vocab_size, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = Seq2SeqModel(
      srce_vocab_size, trgt_vocab_size, _buckets,
      100, 74, 2, 5.0, 1,
      0.01, 0.95, keep_prob=0.8,
      forward_only=forward_only)
  # model.writer = tf.train.SummaryWriter(os.path.join(FLAGS.data_dir, "summary"), session.graph_def) # visualization
  checkpoint_path = os.path.join(data_dir, "checkpoint")
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    raise ValueError(" Invalid arguments! Fails on creating models! ")
  return model


def inter_decode(sent, position, mapp):
  with tf.Session() as sess:
    # Load dictionary
    srce_vocab_path = os.path.join(data_dir, "train", "vocab%d.srce" % 2)
    trgt_vocab_path = os.path.join(data_dir, "train", "vocab%d.trgt" % 0)
    srce_vocab, re_srce_vocab = data_utils.initialize_vocabulary(srce_vocab_path)
    trgt_vocab, re_trgt_vocab = data_utils.initialize_vocabulary(trgt_vocab_path)

    # Create model
    model = create_model(sess, len(re_srce_vocab), len(re_trgt_vocab), True)
    # model.batch_size = 1  # We decode one sentence at a time.

    sentence = sent
    init_pos = eval(position)
    mapp = eval(mapp)

    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(sentence, srce_vocab)
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
    encoder_input, decoder_input, target_weight, pos, maps = model.get_batch(
        {bucket_id: [(token_ids, [], init_pos, mapp)]}, bucket_id)
    # Get output logits for the sentence.
    _, _, output_logits, attentions, env, out_pos = model.step(sess, encoder_input, decoder_input, target_weight, bucket_id, True, 
                decoder_inputs_positions=pos, decoder_inputs_maps=maps)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    
    final_pos = out_pos[0].tolist()
    for l in xrange(len(outputs)-1):
      final_pos.extend(out_pos[l+1].tolist())

    return final_pos