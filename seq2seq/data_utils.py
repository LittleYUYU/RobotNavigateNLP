# Copyright 2015 Google Inc. All Rights Reserved.
# Adapted from TensorFlow released code "data_utils.py".


"""Utilities for process data from different datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# action id
_NO = "0"
_MOVE = "1"
_LEFT = "2"
_RIGHT = "3"
_BACK = "4"
_ACTION_VOCAB = [_NO, _MOVE, _LEFT, _RIGHT, _BACK]

noAct_ID = 4
moveAct_ID = 5
turnLeft_ID = 6
turnRight_ID = 7
turnBack_ID = 8

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    #words.extend(re.split(_WORD_SPLIT, space_separated_fragment)) #we consider simbols other than space
    words.append(space_separated_fragment)
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, min_vocabulary_threshold=0,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary. (===we don't use it===)
    min_vocabulary_threshold: drop words has a frequency less than min_vocabulary_threshold.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s. (default: false)

  Return:
    vocabulary_size: size of the created vocabulary.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 1000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1

      # drop words with frequency less than min_vocabulary_threshold
      for (word, fre) in vocab.items():
        if fre < min_vocabulary_threshold:
          vocab.pop(word)
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      # if len(vocab_list) > max_vocabulary_size:
      #   vocab_list = vocab_list[:max_vocabulary_size]
      
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

    # return size of vocabulary
    print ("Vocabulary size: %s" % len(vocab_list))
    return len(vocab_list)
  # vocabulary exists
  else:
    print ("Vocabulary file %s exists, now use the old one." % vocabulary_path)
    return get_vocab_size(vocabulary_path)
    

def get_vocab_size(vocabulary_path):
  """ Get the size of the vocabulary. """
  size = 0
  with gfile.GFile(vocabulary_path, mode="r") as vocab_file:
    lines = vocab_file.readlines()
    size = len(lines)
    print ("Vocabulary size: %s" % size)
  return size


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s. (default: false)

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def token_ids_to_sentence(token_ids, re_vocab):
  """ Return token ids to original sentence.
  Args:
  token_ids: a list of token ids.
  re_vocab: a list of tokens, serving as a reversed vocabulary.
  """
  sentence = [re_vocab[id] for id in token_ids]
  return " ".join(sentence)

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s. (default: false)
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, srce_vocabulary_min, trgt_vocabulary_min):
  """Get GEO data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    srce_vocabulary_min: threshold for droping less-frequent words in source dictionary.
    trgt_vocabulary_min: threshold for dropping less-frequent words in target dictionary.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for source training data-set,
      (2) path to the token-ids for target training data-set,
      (3) path to the positions of target training data-set,
      (4) path to the map no. of target training data-set,
      (5) path to the token-ids for source development data-set,
      (6) path to the token-ids for target development data-set,
      (7) path to the positions of target development data-set,
      (8) path to the map no. of target development data-set,
      (9) path to the source vocabulary file,
      (10) path to the target vocabluary file.
      (11) source vocabluary size
      (12) target vocabulary size
  """
  # Get wmt data to the specified directory.
  train_path = os.path.join(data_dir, "train")
  dev_path = os.path.join(data_dir, "dev")
  test_path = os.path.join(data_dir, "test")

  # Create vocabularies of the appropriate sizes.
  srce_vocab_path = os.path.join(train_path, "vocab%d.srce" % srce_vocabulary_min)
  trgt_vocab_path = os.path.join(train_path, "vocab%d.trgt" % trgt_vocabulary_min)
  srce_vocab_size = create_vocabulary(srce_vocab_path, train_path + "/data.srce", min_vocabulary_threshold=srce_vocabulary_min)
  # trgt_vocab_size = create_vocabulary(trgt_vocab_path, train_path + "/data.trgt", min_vocabulary_threshold=trgt_vocabulary_min)
  trgt_vocab_size = 9

  # sharing the same vacoabularies for source and target semantic space.
  # Create token ids for the training data.
  srce_train_ids_path = train_path + ("/ids%d.srce" % srce_vocabulary_min)
  trgt_train_ids_path = train_path + ("/ids%d.trgt" % trgt_vocabulary_min)
  data_to_token_ids(train_path + "/data.srce", srce_train_ids_path, srce_vocab_path)
  data_to_token_ids(train_path + "/data.trgt", trgt_train_ids_path, trgt_vocab_path)

  # Create token ids for the development data.
  srce_dev_ids_path = dev_path + ("/ids%d.srce" % srce_vocabulary_min)
  trgt_dev_ids_path = dev_path + ("/ids%d.trgt" % trgt_vocabulary_min)
  data_to_token_ids(dev_path + "/data.srce", srce_dev_ids_path, srce_vocab_path)
  data_to_token_ids(dev_path + "/data.trgt", trgt_dev_ids_path, trgt_vocab_path)

  # positions
  trgt_train_positions_path = os.path.join(train_path, "positions.trgt")
  trgt_train_maps_path = os.path.join(train_path, "map.srce")
  trgt_dev_positions_path = os.path.join(dev_path, "positions.trgt")
  trgt_dev_maps_path = os.path.join(dev_path, "map.srce")

  return (srce_train_ids_path, trgt_train_ids_path, trgt_train_positions_path, trgt_train_maps_path,
          srce_dev_ids_path, trgt_dev_ids_path, trgt_dev_positions_path, trgt_dev_maps_path,
          #srce_test_ids_path, trgt_test_ids_path,
          srce_vocab_path, trgt_vocab_path,
          srce_vocab_size, trgt_vocab_size)


