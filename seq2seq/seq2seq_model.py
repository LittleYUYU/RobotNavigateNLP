# seq2seq_model.py
# Modified based on Tensorflow example code seq2seq_model.py
# Add: environment state and intial position in a map for decoding step by step.
# Fitted to robot navigation task.

"""Sequence-to-sequence model with an attention mechanism, designed for robot navigation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import seq2seq
import seq2seq

import pdb #debug

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/pdf/1601.01280v1.pdf. For training, we adopt
  "dropout" for avoiding overfitting. Trained with AdamOptimizer.

  Update: add summaries for visualization.
  """

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size, state_size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, keep_prob=1.0, forward_only=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      state_size: size of environment representation.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      keep_prob: probability DO NOT dropout.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.state_size = state_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # Create the internal multi-layer cell for our RNN.
    cell = rnn_cell.BasicLSTMCell(size)
    if keep_prob < 1.0 and (not forward_only):
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([cell] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    # define the seq2seq model
    def seq2seq_f(encoder_inputs, decoder_inputs, decoder_inputs_positions, 
      decoder_inputs_maps, do_decode):
      return seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell, source_vocab_size, 
          target_vocab_size, batch_size, self.state_size,
          decoder_inputs_positions=decoder_inputs_positions,
          decoder_inputs_maps=decoder_inputs_maps, feed_previous=do_decode)


    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    self.decoder_inputs_positions = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[self.batch_size],
                                                name="weight{0}".format(i)))
      self.decoder_inputs_positions.append(tf.placeholder(tf.int32, shape=[self.batch_size, 3],
                                                name="position{0}".format(i)))
    
    self.decoder_inputs_maps = tf.placeholder(tf.int32, shape=[self.batch_size], name="mapNo")

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]


    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, self.target_vocab_size,
          lambda x, y, p, m: seq2seq_f(x, y, p, m, True),
          decoder_inputs_positions=self.decoder_inputs_positions, decoder_inputs_maps=self.decoder_inputs_maps)
    else:
      self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, self.target_vocab_size,
          lambda x, y, p, m: seq2seq_f(x, y, p, m, False),
          decoder_inputs_positions=self.decoder_inputs_positions, decoder_inputs_maps=self.decoder_inputs_maps)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      # opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
      opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())
    

  def step(self, session, encoder_inputs, decoder_inputs, target_weights, 
           bucket_id, forward_only, decoder_inputs_positions=None, decoder_inputs_maps=None):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      decoder_inputs_positions: a list of int32 vectors.
      decoder_inputs_maps: a list of int32 numbers.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of enconder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
      input_feed[self.decoder_inputs_positions[l].name] = decoder_inputs_positions[l] if decoder_inputs_positions else None
    
    input_feed[self.decoder_inputs_maps.name] = decoder_inputs_maps
    
    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)

    if not forward_only:
      return outputs[1], outputs[2], None# Gradient norm, loss, no outputs
    else:
      return None, outputs[0], outputs[1:]# No gradient norm, loss, outputs

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      (encoder_inputs, decoder_inputs, target_weights, env_state, initial_position) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    positions = []
    maps = []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input, pos, mapp = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
      pos_pad_size = decoder_size - len(pos)
      positions.append(pos + [pos[-1]] * pos_pad_size)
      maps.append(mapp)

      # pdb.set_trace()

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_decoder_inputs_positions = [], [], [], []
    batch_decoder_inputs_maps = np.array(maps) # 1D int32 Tensor

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    # pdb.set_trace()
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
      batch_decoder_inputs_positions.append(
          np.array([positions[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    # pdb.set_trace()
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_decoder_inputs_positions, batch_decoder_inputs_maps
