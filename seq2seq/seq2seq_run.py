# seq2seq_run.py
# This is a slightly modified version of original TensorFlow sample code "translate.py".
# Fitted to robot navigation task.


"""Training translation models and decoding from them.

For training, please set:
--date_dir = dir_to_the_dataset
--data_name = name_of_the_data

For test, please set:
--decode = True
--data_dir = dir_to_the_dataset(usually the same as the dir in training)
If you manually set srce_vocab_min or trgt_vocab_min in training, please set it again in test.

For self-test, please set:
--self_test = True

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import pdb #debug

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 50,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("state_size", 74, "Size of environment representation.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("srce_vocab_min", 2, "source vocabulary threshold.")
tf.app.flags.DEFINE_integer("trgt_vocab_min", 0, "target vocabulary threshold.")
tf.app.flags.DEFINE_string("data_dir", ".", "Data directory for training.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("epoch", 20, "Set the training iteration.")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep probability for drop out.")
# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
#                             "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding.")
tf.app.flags.DEFINE_boolean("inter_decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("decode_dev", False, "Set to True for decoding development set.")
tf.app.flags.DEFINE_boolean("decode_test", False, "Set to True for decoding test set.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(7, 5), (15, 10), (25, 15), (50, 30)]   # default


def read_data(source_path, target_path, pos_path, map_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
      tokens in srce/trgt should be ids. need preprocess.
    map_path: path to the file of map no..
    pos_path: path to the file of positions of decoder_inputs.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target, env, pos) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with gfile.GFile(source_path, mode="r") as source_file:
    with gfile.GFile(target_path, mode="r") as target_file:
      with gfile.GFile(map_path, mode="r") as map_file:
        with gfile.GFile(pos_path, mode="r") as pos_file:
          source, target, mapp, pos = source_file.readline(), target_file.readline(), map_file.readline(), pos_file.readline()
          counter = 0
          while source and target and mapp and pos and (not max_size or counter < max_size):
            counter += 1
            if counter % 1000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID) # END OD SENTENCE

            pos_ids = eval(pos)
            map_id = int(mapp)

            for bucket_id, (source_size, target_size) in enumerate(_buckets):
              if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids, pos_ids, map_id])
                break
            source, target, mapp, pos = source_file.readline(), target_file.readline(), map_file.readline(), pos_file.readline()
  return data_set


def create_model(session, srce_vocab_size, trgt_vocab_size, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = Seq2SeqModel(
      srce_vocab_size, trgt_vocab_size, _buckets,
      FLAGS.size, FLAGS.state_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, keep_prob=FLAGS.keep_prob,
      forward_only=forward_only)
  # model.writer = tf.train.SummaryWriter(os.path.join(FLAGS.data_dir, "summary"), session.graph_def) # visualization
  checkpoint_path = os.path.join(FLAGS.data_dir, "checkpoint")
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  """Train a nl -> machine-code translation model."""
  # Prepare training & dev data.
  print("Preparing data in %s" % FLAGS.data_dir)
  srce_train, trgt_train, trgt_train_pos, trgt_train_map, srce_dev, trgt_dev, trgt_dev_pos, trgt_dev_map, _, _, srce_vocab_size, trgt_vocab_size = data_utils.prepare_data(
      FLAGS.data_dir, FLAGS.srce_vocab_min, FLAGS.trgt_vocab_min)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, srce_vocab_size, trgt_vocab_size, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(srce_dev, trgt_dev, trgt_dev_pos, trgt_dev_map)
    train_set = read_data(srce_train, trgt_train, trgt_train_pos, trgt_train_map, max_size=FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print("training set bucket: ", train_bucket_sizes)

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    # size of dev set
    dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
    dev_size = float(sum(dev_bucket_sizes)) 
    dev_bucket_proportion = [b/dev_size for b in dev_bucket_sizes]# proportion
    print("dev set bucket: ", dev_bucket_sizes)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    dev_losses = [] 
    steps_per_checkpoint = int(train_total_size / FLAGS.batch_size)
    print ("steps per checkpoint: ", steps_per_checkpoint)

    while current_step < (FLAGS.epoch * steps_per_checkpoint):
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # check for empty bucket
      if len(train_set[bucket_id]) == 0:
        continue

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights, pos, maps = model.get_batch(
          train_set, bucket_id)

      # step
      _, step_loss, _, _, _, _= model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False, 
                                   decoder_inputs_positions=pos, decoder_inputs_maps=maps)
      step_time += (time.time() - start_time) / steps_per_checkpoint
      loss += step_loss / steps_per_checkpoint
      current_step += 1


      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        # Decrease learning rate if no improvement was seen over last 3 times.
        # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
        if current_step / steps_per_checkpoint > 5:
          sess.run(model.learning_rate_decay_op)
          print ("learning rate update to %.4f" % model.learning_rate.eval())
          if model.learning_rate == float(0):
            break
        previous_losses.append(loss)

        # Run evals on development set, print their perplexity and perform early stopping.
        eval_loss_per_bucket = [] # eval_loss for the whole dev set
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id])==0:
            # print ("Bucket %s is empty." % bucket_id)
            eval_loss_per_bucket.append(float(0))
            continue
          
          encoder_inputs, decoder_inputs, target_weights, pos, maps = model.get_batch(
              dev_set, bucket_id)
          
          _, eval_loss, _, _, _, _= model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True, decoder_inputs_positions=pos, decoder_inputs_maps=maps)
          
          eval_loss_per_bucket.append(float(eval_loss)) 
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_loss))
        
        dev_loss = np.dot(np.asarray(eval_loss_per_bucket), np.asarray(dev_bucket_proportion))
        dev_losses.append(dev_loss)
        print(" eval: dev set weighted perplexity %.2f"% dev_loss)
        
        if dev_loss <= min(dev_losses):
          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(FLAGS.data_dir, "checkpoint/ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        
        step_time, loss = 0.0, 0.0
        
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # load dictionary
    srce_vocab_path = os.path.join(FLAGS.data_dir, "train", "vocab%d.srce" % FLAGS.srce_vocab_min)
    trgt_vocab_path = os.path.join(FLAGS.data_dir, "train", "vocab%d.trgt" % FLAGS.trgt_vocab_min)
    
    _, re_srce_vocab = data_utils.initialize_vocabulary(srce_vocab_path)
    _, re_trgt_vocab = data_utils.initialize_vocabulary(trgt_vocab_path)

    # Load test data.
    if FLAGS.decode_test:
      srce_test_ids_path = os.path.join(FLAGS.data_dir, "test", "ids%d.srce" % FLAGS.srce_vocab_min)
      trgt_test_ids_path = os.path.join(FLAGS.data_dir, "test", "ids.trgt")
      srce_test_data_path = os.path.join(FLAGS.data_dir, "test/data.srce")
      trgt_test_data_path = os.path.join(FLAGS.data_dir, "test/data.trgt")

      # Prepare test data
      data_utils.data_to_token_ids(srce_test_data_path, srce_test_ids_path, srce_vocab_path)
      data_utils.data_to_token_ids(trgt_test_data_path, trgt_test_ids_path, trgt_vocab_path)
      trgt_test_pos = os.path.join(FLAGS.data_dir, "test", "positions.trgt")
      trgt_test_map = os.path.join(FLAGS.data_dir, "test", "map.srce")
      test_set = read_data(srce_test_ids_path, trgt_test_ids_path, trgt_test_pos, trgt_test_map)

    elif FLAGS.decode_dev:
      srce_dev_ids_path = os.path.join(FLAGS.data_dir, "dev", "ids%d.srce" % FLAGS.srce_vocab_min)
      trgt_dev_ids_path = os.path.join(FLAGS.data_dir, "dev", "ids%d.trgt" % FLAGS.trgt_vocab_min)
      trgt_dev_pos = os.path.join(FLAGS.data_dir, "dev", "positions.trgt")
      trgt_dev_map = os.path.join(FLAGS.data_dir, "dev", "map.srce")
      test_set = read_data(srce_dev_ids_path, trgt_dev_ids_path, trgt_dev_pos, trgt_dev_map)

    else:
      raise ValueError(" Please set decode_test or decode_dev to True! ")

    # Create model and load parameters.
    model = create_model(sess, len(re_srce_vocab), len(re_trgt_vocab), True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode test data.  ---> read from files

    decode_result_path = os.path.join(FLAGS.data_dir, ("result/result_size%d_dropout%.2f" % (FLAGS.size, FLAGS.keep_prob)))
    decode_data_path = os.path.join(FLAGS.data_dir, ("result/gold_size%d_dropout%.2f" % (FLAGS.size, FLAGS.keep_prob)))
    
    test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
    print ("test bucket size: ", test_bucket_sizes)

    count = 0
    correct = 0

    with open(decode_result_path, 'w') as fpred:
      with open(decode_data_path, 'w') as fgold: # note that the test data has been sorted by bucket size
        for b in xrange(len(_buckets)):
          print ("bucket%d:" % b)
          
          if len(test_set[b]) == 0: # empty bucket
            continue
          
          for sent in test_set[b]:
            
            encoder_input, decoder_input, target_weight, pos, maps = model.get_batch({b: [sent]}, b)
            # get output_logits
            _, _, output_logits, _, _, _= model.step(sess, encoder_input, decoder_input, target_weight, b, True, 
                  decoder_inputs_positions=pos, decoder_inputs_maps=maps)
            # greedy decoder: outputs are argmax of output_logits
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
              outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            # write to file
            fpred.write(data_utils.token_ids_to_sentence(outputs, re_trgt_vocab) + '\n')
            gold = sent[1]
            if data_utils.EOS_ID in sent[1]:
              gold = sent[1][:sent[1].index(data_utils.EOS_ID)]
            fgold.write(data_utils.token_ids_to_sentence(gold, re_trgt_vocab) + '\n')

            if gold == outputs:
              correct += 1
            # else:
            #   print ("source: ", data_utils.token_ids_to_sentence(sent[0], re_srce_vocab), '\t', pos, '\t', maps)
            #   print ("target: ", data_utils.token_ids_to_sentence(gold, re_trgt_vocab))
            #   print ("predict: ", data_utils.token_ids_to_sentence(outputs, re_trgt_vocab) + '\n')

            count += 1
    print("count = %d, correct = %d, accuracy = %f" % (count, correct, float(correct)/count))

def inter_decode():
  with tf.Session() as sess:
    # Load dictionary
    srce_vocab_path = os.path.join(FLAGS.data_dir, "train", "vocab%d.srce" % FLAGS.srce_vocab_min)
    trgt_vocab_path = os.path.join(FLAGS.data_dir, "train", "vocab%d.trgt" % FLAGS.trgt_vocab_min)
    srce_vocab, re_srce_vocab = data_utils.initialize_vocabulary(srce_vocab_path)
    trgt_vocab, re_trgt_vocab = data_utils.initialize_vocabulary(trgt_vocab_path)

    # Create model
    model = create_model(sess, len(re_srce_vocab), len(re_trgt_vocab), True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode from standard input.  ---> interactive decoding
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # read supplement input: children, weight.
      init_pos = eval(sys.stdin.readline())
      mapp = eval(sys.stdin.readline())
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(sentence, srce_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      # pdb.set_trace()
      encoder_input, decoder_input, target_weight, pos, maps = model.get_batch(
          {bucket_id: [(token_ids, [], init_pos, mapp)]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits, attentions, env, out_pos = model.step(sess, encoder_input, decoder_input, target_weight, bucket_id, True, 
                  decoder_inputs_positions=pos, decoder_inputs_maps=maps)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      
      # If there is an EOS symbol in outputs, cut them at that point.
      # if data_utils.EOS_ID in outputs:
      #   outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      

      # print predicted result
      print ("predict: ", data_utils.token_ids_to_sentence(outputs, re_trgt_vocab))
      for l in xrange(len(outputs)):
        print (l, '\t', re_trgt_vocab[outputs[l]])
        print ("attention weight: ", attentions[l])
        print ("environment: ", env[l])
        print ("output position: ", out_pos[l], '\n')

      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  # run the model
  with tf.Session() as sess:
    # pdb.set_trace()
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = Seq2SeqModel(10, 9, [(3, 3), (6, 6)], 32, 74, 2,
                                       5.0, 5, 0.3, 0.95)
    model.writer = tf.train.SummaryWriter("./summary", graph_def=sess.graph_def)
    # run the session
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2], [[1,1,0], [1,2,0], [1,2,1]], 0), ([3, 3], [4], [[0,5,2],[0,5,0]], 1), ([5], [2],
     [[6,7,3], [6,6,3]], 2)], [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [[5,5,0],[5,6,0],[5,7,0],[5,8,0],[5,9,0],[5,10,0]], 1), ([3, 3, 3], [1, 3], [[2,2,1],[2,2,1],[2,2,2]], 2)])
                

    for _ in xrange(5):  # Train the fake model for 5 epochs.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights, positions, maps = model.get_batch(
          data_set, bucket_id)
      _, loss, _, _, _= model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False, decoder_inputs_positions=positions, 
        decoder_inputs_maps=maps)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.inter_decode:
    inter_decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
