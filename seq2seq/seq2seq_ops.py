# seq2seq_ops.py
# Library for creating sequence-to-sequence models.
# Modified based on Tensorflow source code.
# ==============================================================================

"""Library for creating sequence-to-sequence models, especially for navigating 
robots to follos an instruction in a map.
This file has been modified to maintain only the embedding_attention_seq2seq model.
The attention mechanism refers to http://arxiv.org/pdf/1601.01280v1.pdf.
In addition, we add an environment representation in decoder to incorporate 
environment information in a map."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs

import map
import data_utils

import pdb # debug


def attention_decoder(decoder_inputs, initial_state, attention_states, cell, batch_size, state_size,
                      decoder_inputs_positions=None, decoder_inputs_maps=None, output_size=None, loop_function=None,
                      dtype=dtypes.float32, scope=None):
  """RNN decoder with attention for the sequence-to-sequence model.

  Args:
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size]. Embedded inputs.
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    batch_size: need to clarify batch size explicitly since env_state is updated one sample by one sample.
    state_size: size of environment state.
    decoder_inputs_positions: a list of 2D Tensors of shape [batch_size, 3],
       indicating intial positions of each example in a map. Default None.
    decoder_inputs_maps: a 1D Tensor of length batch_size indicating the map. Default None.
    output_size: size of the output vectors; if None, we use cell.output_size.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
      [batch_size x output_size]. These represent the generated outputs.
      Output i is computed from input i (which is either i-th decoder_inputs or
      loop_function(output {i-1}, i)) as follows. 
      First, we run the cell on the current decoder input or feed from previous output:
        cur_output, new_state = cell(input, prev_state).
      Then, we calculate new attention masks:
        new_attn = softmax(h_t^T * attention_states).
      Thus, the context vector:
        cont_vec = weighted_sum_of(attention_states), weighted by (new_attn),
      and then we calculate the attended output:
        attn_output = tanh(W1*current_output + W2*cont_vec + W3*env_state).
      The finally output for prediction:
        output = softmax(W*attn_output).
        This "output" should be a 1D Tensor of shape [num_symbols].
        Every item of the output refers to the probability of predicting certain symbol for the next step.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with vs.variable_scope(scope or "attention_decoder"):
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    mapIdx = array_ops.pack([map.map_grid, map.map_jelly, map.map_one]) #map

    attention_vec_size = attn_size # size of query
    states = [initial_state]
    # current position and environment
    position, env = None, None

    hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size]) # reshape for later computation

    def attention(query): 
      """Put attention masks on hidden using hidden_features and query."""
      with vs.variable_scope("Attention"):
        # Attention mask is a softmax of h_in^T*decoder_hidden.
        dec_hid = array_ops.tile(query, [1, attn_length]) # replicate query for element-wise multiplication
        dec_hid = array_ops.reshape(dec_hid, [-1, attn_length, attention_vec_size])
        attn_weight = nn_ops.softmax(math_ops.reduce_sum(attention_states*dec_hid, [2])) # attn weights for every hidden states in encoder
        # Now calculate the attention-weighted vector (context vector) cc.
        cc = math_ops.reduce_sum(array_ops.reshape(attn_weight, [-1, attn_length, 1, 1])*hidden, [1,2])
        # attented hidden state
        with vs.variable_scope("AttnW1"):
          term1 = rnn_cell.linear(query, attn_size, False)
        with vs.variable_scope("AttnW2"):
          term2 = rnn_cell.linear(cc, attn_size, False)
        # pdb.set_trace()
        # environment representation
        if env: # 2D Tensor of shape [batch_size, env_size]
          with vs.variable_scope("Environment"):
            term3 = rnn_cell.linear(math_ops.to_float(env), attn_size, False)
          h_attn = math_ops.tanh(term1 + term2 + term3)
        else:
          h_attn = math_ops.tanh(term1 + term2)
      return h_attn


    def updateEnv(position, step, mapNo):
      """ Update env_state according to current position and step.
      Args:
      position: a 2D Tensor of shape [batch_size, 3].
      step: a 2D Tensor of shape [batch_size, 1], where
      0 --> no action, 1 --> move forward 1 step, 2 --> turn right, 3 --> turn left, 4 --> turn back.
      mapNo: a 1D int32 Tensor of length batch_size.
      
      Returns:
      env: a 2D Tensor of shape [batch_size, env_size]
        environment state after taking the step based on the position.
      position: a 2D Tensor of shape [batch_size, 3]
        new position after taking the step based on the position.
      """
      # pdb.set_trace()
      if not mapNo:
        raise ValueError(" Invalid argument mapNo in updateEnv! ")
      if not position:
        raise ValueError(" Invalid argument position in updateEnv! ")
      new_env = []
      new_pos = []
      # if step == None, take no step and return the environment representations of each position.
      if not step:
        new_pos = position 
        for j in xrange(batch_size):
          new_env.append(array_ops.reshape(
            array_ops.slice(mapIdx, array_ops.pack([mapNo[j], position[j,0], position[j,1], position[j,2], 0]), [1,1,1,1,state_size]), [state_size]))
          # new_env.append(array_ops.reshape(
          #   array_ops.slice(mapIdx, array_ops.pack([mapNo[j], position[j][0], position[j][1], position[j][2], 0]), [1,1,1,1,-1]), [state_size]))
        new_env = array_ops.reshape(array_ops.pack(new_env), [batch_size, state_size])
        return new_pos, new_env
      else:
        for j in xrange(batch_size):
          if step[j,0] == np.int32(data_utils.noAct_ID): # no action
            new_pos.append(position[j,:])
          
          elif step[j,0] == np.int32(data_utils.moveAct_ID): # move forward 1 step
            if position[j,2] == np.int32(0): # 0
              new_pos.append(position[j,:] + np.array([1, 0, 0]))
            elif position[j,2] == np.int32(1): # 90
              new_pos.append(position[j,:] + np.array([0, 1, 0]))
            elif position[j,2] == np.int32(2): # 180
              new_pos.append(position[j,:] + np.array([-1, 0, 0]))
            else: # 270
              new_pos.append(position[j,:] + np.array([0, -1, 0]))
            
          elif step[j,0] == np.int32(data_utils.turnRight_ID): # turn right
            if position[j,2] == np.int32(0): # direction 0 --> 270
              new_pos.append(array_ops.pack([position[j,0], position[j,1], np.int32(3)]))
            else:
              new_pos.append(position[j,:] + np.array([0, 0, -1]))
          
          elif step[j,0] == np.int32(data_utils.turnLeft_ID): # turn left
            if position[j,2] == np.int32(3):
              new_pos.append(array_ops.pack([position[j,0], position[j,1], np.int32(0)]))
            else:
              new_pos.append(position[j,:] + np.array([0, 0, 1]))
          
          else: # turn back
            if position[j,2] == np.int32(2):
              new_pos.append(array_ops.pack([position[j,0], position[j,1], np.int32(0)]))
            elif position[j,2] == np.int32(3):
              new_pos.append(array_ops.pack([position[j,0], position[j,1], np.int32(1)]))
            else:
              new_pos.append(position[j,:] + np.array([0, 0, 2]))

          # update environment
          new_env.append(array_ops.reshape(
            array_ops.slice(mapIdx, array_ops.pack([mapNo[j], new_pos[-1][0], new_pos[-1][1], new_pos[-1][2], 0]), [1,1,1,1,-1]),[state_size]))
        
        new_pos = array_ops.pack(new_pos)
        new_env = array_ops.pack(new_env)
        return new_pos, new_env

    # pdb.set_trace()
    outputs = []
    prev = None
    if decoder_inputs_positions and decoder_inputs_maps and batch_size:
      position = decoder_inputs_positions[0] # 2d tensor of shape [batch_size, 3]
      _, env = updateEnv(position, None, decoder_inputs_maps)
    for i in xrange(len(decoder_inputs)):
      if i > 0:
        vs.get_variable_scope().reuse_variables()
      inp = decoder_inputs[i]
      
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with vs.variable_scope("loop_function", reuse=True):
          inp = array_ops.stop_gradient(loop_function(prev, i))

      # Run the RNN.
      cur_output, new_state = cell(inp, states[-1])
      states.append(new_state)

      # Run the attention mechanism.
      h_attn = attention(cur_output)
      
      with vs.variable_scope("AttnOutputProjection"):
        output = rnn_cell.linear(h_attn, output_size, False)
      
      if loop_function is not None:
        # We do not propagate gradients over the loop function.
        prev = array_ops.stop_gradient(output)
      
      if decoder_inputs_positions[0] and decoder_inputs_maps and position:
        # update current environment
        if loop_function is not None:
          step = math_ops.argmax(nn_ops.softmax(prev), 1) # step is a list (len=batch_size) of int32 number
          position, env = updateEnv(position, step, decoder_inputs_maps)
        else:
          position = decoder_inputs_positions[i+1] if (i < len(decoder_inputs_positions) - 1) else position
          _, env = updateEnv(position, None, decoder_inputs_maps)

      outputs.append(output)

  return outputs, states


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, batch_size, state_size, decoder_inputs_positions=None,
                                decoder_inputs_maps=None, output_size=None, feed_previous=False,
                                dtype=dtypes.float32,
                                scope=None):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: a list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: integer, how many symbols come into the embedding.
    batch_size: need to clarify for decoding.
    decoder_inputs_positions: a list of 2D Tensors of shape [batch_size, 3].
    decoder_inputs_maps: a 1D Tensor of length batch_size.
    output_size: size of the output vectors; if None, use cell.output_size.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when output_projection has the wrong shape.

  Modification:
    No output projection.
  """
  if output_size is None:
    output_size = cell.output_size

  with vs.variable_scope(scope or "embedding_attention_decoder"):
    with ops.device("/cpu:0"):
      embedding = vs.get_variable("embedding", shape=[num_symbols, cell.input_size], 
        initializer=init_ops.random_uniform_initializer(-0.08, 0.08))

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
      emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, batch_size, state_size,
        decoder_inputs_positions=decoder_inputs_positions, decoder_inputs_maps=decoder_inputs_maps, output_size=output_size,
        loop_function=loop_function)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                num_encoder_symbols, num_decoder_symbols, batch_size, state_size,
                                decoder_inputs_positions=None, decoder_inputs_maps=None, feed_previous=False, 
                                dtype=dtypes.float32,
                                scope=None):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x cell.input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  cell.input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Args:
    encoder_inputs: a list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: a list of 1D int32 Tensors of shape [batch_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: integer; number of symbols on the encoder side.
    num_decoder_symbols: integer; number of symbols on the decoder side.
    batch_size: need to clarify for decoding.
    decoder_inputs_positions: a list of 2D Tensors of shape [batch_size, 3].
    decoder_inputs_maps: a 1D Tensor of length batch_size.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x num_decoder_symbols] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  
  Modification:
    No output projection wrapper used compared to the original version.
  """
  with vs.variable_scope(scope or "embedding_attention_seq2seq"):
    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(cell, num_encoder_symbols, 
      initializer=init_ops.random_uniform_initializer(-0.08, 0.08))
    encoder_outputs, encoder_states = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs, encoder_states[-1], attention_states, cell,
          num_decoder_symbols, batch_size, state_size, decoder_inputs_positions=decoder_inputs_positions,
          decoder_inputs_maps=decoder_inputs_maps, output_size=output_size,
          feed_previous=feed_previous)
    else:  # If feed_previous is a Tensor, we construct 2 graphs and use cond. 
      # We don't consider this case.
      raise ValueError("Imcompatible variable feed_previous.\n")



def sequence_loss_by_example(logits, targets, weights, num_decoder_symbols,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: list of 1D batch-sized int32 Tensors of the same length as logits.
    weights: list of 1D batch-sized float-Tensors of the same length as logits.
    num_decoder_symbols: integer, number of decoder symbols (output classes).
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: the log-perplexity for each sequence.

  Raises:
    ValueError: if len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.op_scope(logits + targets + weights, name,
                    "sequence_loss_by_example"):
    batch_size = array_ops.shape(targets[0])[0]
    log_perp_list = []
    length = batch_size * num_decoder_symbols
    for i in xrange(len(logits)):
      if softmax_loss_function is None:
        # TODO(lukaszkaiser): There is no SparseCrossEntropy in TensorFlow, so
        # we need to first cast targets into a dense representation, and as
        # SparseToDense does not accept batched inputs, we need to do this by
        # re-indexing and re-sizing. When TensorFlow adds SparseCrossEntropy,
        # rewrite this method.
        indices = targets[i] + num_decoder_symbols * math_ops.range(batch_size)
        with ops.device("/cpu:0"):  # Sparse-to-dense must be on CPU for now.
          dense = sparse_ops.sparse_to_dense(
              indices, array_ops.expand_dims(length, 0), 1.0,
              0.0)
        target = array_ops.reshape(dense, [-1, num_decoder_symbols])
        crossent = nn_ops.softmax_cross_entropy_with_logits(
            logits[i], target, name="SequenceLoss/CrossEntropy{0}".format(i))
      else:
        crossent = softmax_loss_function(logits[i], targets[i])
      log_perp_list.append(crossent * weights[i])
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights, num_decoder_symbols,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: list of 2D Tensors os shape [batch_size x num_decoder_symbols].
    targets: list of 1D batch-sized int32 Tensors of the same length as logits.
    weights: list of 1D batch-sized float-Tensors of the same length as logits.
    num_decoder_symbols: integer, number of decoder symbols (output classes).
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: the average log-perplexity per symbol (weighted).

  Raises:
    ValueError: if len(logits) is different from len(targets) or len(weights).
  """
  with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights, num_decoder_symbols,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, num_decoder_symbols, seq2seq, 
                       decoder_inputs_positions=None, decoder_inputs_maps=None,
                       softmax_loss_function=None, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model, e.g., 
  seq2seq = lambda x, y, p, m: basic_rnn_seq2seq(x, y, decoder_inputs_positions=p, 
    decoder_inputs_maps=m, feed_previous=True/False)

  Args:
    encoder_inputs: a list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: a list of Tensors to feed the decoder; second seq2seq input.
    targets: a list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: list of 1D batch-sized float-Tensors to weight the targets.
    buckets: a list of pairs of (input size, output size) for each bucket.
    num_decoder_symbols: integer, number of decoder symbols (output classes).
    seq2seq: a sequence-to-sequence model function; it takes 4 input that
      agree with encoder_inputs, decoder_inputs, environment state and initial position in the map,
      and returns a pair consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    decoder_inputs_positions: a list of 2D Tensors of shape [batch_size, 3].
    decoder_inputs_maps: a 1D Tensor of length batch_size.
    softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: optional name for this operation, defaults to "model_with_buckets".

  Returns:
    outputs: The outputs for each bucket. Its j'th element consists of a list
      of 2D Tensors of shape [batch_size x num_decoder_symbols] (j'th outputs).
    losses: List of scalar Tensors, representing losses for each bucket.
  Raises:
    ValueError: if length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  if decoder_inputs_positions and decoder_inputs_maps:
    all_inputs += decoder_inputs_positions
    all_inputs.append(decoder_inputs_maps)

  losses = []
  outputs = []
  with ops.op_scope(all_inputs, name, "model_with_buckets"):
    for j in xrange(len(buckets)):
      if j > 0:
        vs.get_variable_scope().reuse_variables()
      bucket_encoder_inputs = [encoder_inputs[i]
                               for i in xrange(buckets[j][0])]
      bucket_decoder_inputs = [decoder_inputs[i]
                               for i in xrange(buckets[j][1])]
      bucket_decoder_inputs_positions = [decoder_inputs_positions[i]
                                for i in xrange(buckets[j][1])]
      bucket_decoder_inputs_maps = decoder_inputs_maps
      
      # pdb.set_trace() #debug
      bucket_outputs, _ = seq2seq(bucket_encoder_inputs,
                                  bucket_decoder_inputs, 
                                  bucket_decoder_inputs_positions,
                                  bucket_decoder_inputs_maps)
      outputs.append(bucket_outputs)

      bucket_targets = [targets[i] for i in xrange(buckets[j][1])]
      bucket_weights = [weights[i] for i in xrange(buckets[j][1])]
      
      # pdb.set_trace() #debug
      losses.append(sequence_loss(
          outputs[-1], bucket_targets, bucket_weights, num_decoder_symbols,
          softmax_loss_function=softmax_loss_function))

  return outputs, losses
