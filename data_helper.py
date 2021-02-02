"""
Author: Wheeler Earnest

"""

import numpy as np
import pandas as pd
import torch
import gensim

from collections import Counter
from torch.utils.data import Dataset

EMPTY_LINE = 'EMPTY'
COMMENT_ONLY_LINE = 'COMMENTEMPTY'
UNKNOWN_WORD = "-unk-"
SPECIAL_SYMBOLS = [UNKNOWN_WORD]


def read_comment_files(data_file, max_line_length):
    """
    Parameters
    -----------
    data_file : str
        Path to the file containing the comment location data

    max_line_length : int
        Maximum number of tokens per line that will be read in

    Returns
    ----------
    files_of_code : list of FileOfCode
        List of all files, each of which contain code blocks

    all_words : Counter
        Counter keeping track of the occurrences of each word

    """
    d_file = open(data_file, 'r')
    line_id = -1

    all_words = Counter()
    current_file_contents = FileOfCode()

    # List containing the loc data of each file as a file_of_code object
    files_of_code = []

    current_file_id = ""

    for line in d_file:
        # Increment the line_id and pull data from the text file
        line_id += 1
        file_id, line_number, block_bnd, label, _, _, code, _, _ = line.strip().split("\t")

        if code == COMMENT_ONLY_LINE or EMPTY_LINE:
            continue

        # Clean up the code and count the words
        code_words = code.split()[0:min(len(code.split()), max_line_length)]
        for word in code_words:
            all_words[word] += 1

        code = " ".join(code_words).strip()

        current_loc = LineOfCode(file_id, line_number, block_bnd, code, label)

        # Check to see if this loc is in the same file as the previous
        if file_id != current_file_id:

            if current_file_contents.len() > 0:
                files_of_code.append(current_file_contents)
                current_file_contents = FileOfCode()

            current_file_id = file_id

        current_file_contents.add_loc(current_loc)

    if current_file_contents.len() > 0:
        files_of_code.append(current_file_contents)

    d_file.close()

    return files_of_code, all_words


def read_embeddings(embeddings_path):
    """
    Parameters
    ----------
    embeddings_path: str
        Path to where the word2vec model is stored

    Returns
    -------
    code_embeds
        w2v model

    code_embeds_dim
        dimensions of the w2v model
    """
    code_embeds = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    comm_embeds = None

    code_embeds_dim = code_embeds.vector_size
    comm_embeds_dim = None

    return code_embeds, code_embeds_dim


def create_vocabularies(word_counts, vocab_size, min_word_freq):
    """

    Parameters
    ----------
    word_counts: Counter
        Counts the occurrences each word
    vocab_size: int
        Limit the size of the vocabulary
    min_word_freq: int
        Minimum occurrences a word must have to be included


    Returns
    -------
    words: list
        list of all the words in the vocabulary
    word_w2i: dict
        the word is the key, and the id is the value
    word_i2w: dict
        reverse of the other
    """
    size_without_specials = min(len(word_counts), vocab_size)
    if size_without_specials == vocab_size:
        size_without_specials -= len(SPECIAL_SYMBOLS)
    # Sort the word counts by occurrence then alphabetically
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    # Throw out words below frequency threshold and limit the vocab size
    words = [k for (k, v) in sorted_word_counts if k not in SPECIAL_SYMBOLS and v >= min_word_freq][
            0:size_without_specials]
    for s in SPECIAL_SYMBOLS:
        words.insert(0, s)
    word_w2i, word_i2w = assign_vocab_ids(words)

    return words, word_w2i, word_i2w



def assign_vocab_ids(words):
    """

    Parameters
    ----------
    words

    Returns
    -------
    word_to_id: dict
        Dictionary with ids of each word and vice versa
    id_to_word: dict
        Dictionary with words of each id

    """
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(words, range(len(words))))
    return word_to_id, id_to_word



def create_dataset(data_file, max_len_stmt, pretrained_embeddings_path):
    data, words = read_comment_files(data_file, 100)

    embeddings, embeddings_dim = read_embeddings(pretrained_embeddings_path)


class LineOfCode(object):
    """
    Contains all the information about a specific line of code.
    """

    def __init__(self, file_id, line_number, block_bnd, code, label):
        """
        Parameters
        ----------------
        file_id : int

        line_number: int

        block_bnd: int
             -1 if NA, 1 if start of code block, 2 if middle, and 3 if end

        code: string
            Words on the line of code

        label: int
            0/1, Tells if a comment appears before this code

        """

        self.file_id = file_id
        self.line_number = line_number
        self.block_bnd = block_bnd
        self.code = code
        self.label = label


class FileOfCode(object):
    """
    Contains all the lines of code for an individual file
    """

    def __init__(self):
        self.locs = []

    def add_loc(self, loc):
        self.locs.append(loc)

    def len(self):
        return len(self.locs)

    def get_loc(self, idx):
        return self.locs[idx]


class CommentDataset(Dataset):

    def __init__(self, files_of_code):
        self.files = files_of_code

    def __getitem__(self, index):
        return self.files[index]
