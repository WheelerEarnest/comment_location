"""
Author: Wheeler Earnest

"""

import numpy as np
import pandas as pd
import torch
import gensim
from gensim.models import KeyedVectors
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

    code_embeds_dim: int
        dimensions of the model vectors
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
    words: list of str
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


def prepare_loc_data(file_seqs, max_loc_len, vocab, words_w2i, words_i2w, w2v_models, w2v_dims):
    """

    Parameters
    ----------
    file_seqs: list of LineOfCode objects
    max_loc_len: int
        max length of a line of code
    vocab: list of str
        all the words in the vocabulary
    words_w2i: dict
    words_i2w: dict
    w2v_models
    w2v_dims

    Returns
    -------
    loc_id_to_coded: dict
        Key format is: "fileid#line_number"
        value is the coded loc
    """
    loc_id_to_coded = {}

    for file in file_seqs:
        for loc_num in range(len(file)):
            loc = file.get_loc(loc_num)
            loc_code = loc.code

            loc_encoded = encode_words(loc_code, words_w2i, False)
            loc_embedded = get_avg_embedding(loc_encoded, words_i2w, w2v_models, w2v_dims)

            loc_id_to_coded[str(loc.file_id) + "#" + str(loc.line_number)] = loc_encoded

    return loc_id_to_coded


def encode_words(words, words_w2i, ignore_unknown):
    encoded = []
    # Loop through the words and append if found in vocabulary
    for w in words:
        if w in words_w2i:
            encoded.append(words_w2i[w])
        elif not ignore_unknown:
            encoded.append(words_w2i[UNKNOWN_WORD])
    return encoded


def get_avg_embedding(word_ids, words_i2w, w2v_model, w2v_dim):
    """
    Averages the embeddings for a line of code. If w2v_model is none, then a zero vector is returned.
    Parameters
    ----------
    word_ids
    words_i2w
    w2v_model
    w2v_dim

    Returns
    -------
    word_sum : numpy.ndarray
    """

    word_sum = np.zeros(w2v_dim, dtype=np.float32)
    if w2v_model == None:
        return word_sum
    word_count = 0
    for wid in word_ids:
        word = words_i2w[wid]
        if word != UNKNOWN_WORD and word in w2v_model.vocab:
            word_sum += w2v_model.wv[word]
            word_count += 1
    if word_count > 0:
        return word_sum / word_count
    return word_sum


def create_dataset(data_file, max_len_stmt, pretrained_embeddings_path, max_vocab_size, min_word_freq):
    data, words = read_comment_files(data_file, 100)

    embeddings, embeddings_dim = read_embeddings(pretrained_embeddings_path)
    vocab, vocab_w2i, vocab_i2w = create_vocabularies(words, max_vocab_size, min_word_freq)

    loc_id_to_coded = prepare_loc_data(data, max_len_stmt, vocab, vocab_w2i, vocab_i2w, embeddings, embeddings_dim)


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

    def __init__(self, files_of_code, loc_id_to_coded,
                 vocab, vocab_w2i, max_blocks, max_locs, max_words):
        """

        Parameters
        ----------
        files_of_code
        loc_id_to_coded
        vocab
        vocab_w2i
        max_blocks: int
            max blocks per sequence
        max_locs: int
            max locs per block
        max_words: int
            max words per loc
        """
        self.files = files_of_code
        self.loc_id_to_coded = loc_id_to_coded
        self.vocab = vocab
        self.vocab_w2i = vocab_w2i
        self.max_blocks = max_blocks
        self.max_locs = max_locs
        self.max_words = max_words
        self.data, self.data_weights, self.labels, self.label_weights = self._prepare_batch_data()

    def __getitem__(self, index):
        return self.files[index]

    def _prepare_batch_data(self):
        """

        Returns
        -------
        data: tensor int32
        data_weights: tensor float32
        labels: tensor int32
        label_weights: tensor float32
        """
        data_by_seq = []
        cur_seq = []
        count_true_blks = 0

        # Loop through our files and arrange blocks sequentially
        for file in self.files:
            if len(data_by_seq) == 0 or data_by_seq[-1] != []:
                data_by_seq.append([])

            # Loop through each loc and group by block boundary
            for loc_num in range(len(file)):
                loc = file.get_loc(loc_num)
                loc_coded = self.loc_id_to_coded[str(loc.file_id) + "#" + str(loc.line_number)]
                loc_bnd = loc.block_bnd

                if loc_bnd == 1:
                    if len(cur_seq) > 0:
                        data_by_seq[-1].append(cur_seq)
                        cur_seq = []

                    # If we're past the max blocks then create new sequence
                    if len(data_by_seq[-1]) >= self.max_blocks:
                        data_by_seq.append([])
                    count_true_blks += 1
                    cur_seq.append(loc)

                elif loc_bnd == 2 or loc_bnd == 3:
                    if len(cur_seq) < self.max_locs:
                        cur_seq.append(loc)

            # If we the end of a file and the current sequence is not empty, append it and start a new sequence
            if len(cur_seq) > 0:
                data_by_seq[-1].append(cur_seq)
                cur_seq = []

            count_blocks = 0
            for i in range(len(data_by_seq)):
                for j in range(len(data_by_seq[i])):
                    count_blocks += 1

        assert count_true_blks == count_blocks, "Number of blocks read %d doesn't match true number of blocks %d" % (
            count_blocks, count_true_blks)

        total_seqs = len(data_by_seq)
        data = np.zeros((total_seqs, self.max_blocks, self.max_locs, self.max_words), dtype=np.int32)
        data_wts = np.zeros((total_seqs, self.max_blocks, self.max_locs, self.max_words), dtype=np.float32)
        labels = np.zeros((total_seqs, self.max_blocks), dtype=np.int32)
        label_weights = np.zeros((total_seqs, self.max_blocks), dtype=np.float32)

        for i in range(total_seqs):
            for j in range(self.max_blocks):
                for k in range(self.max_locs):
                    # Check to see if we have moved past the sequence data, and if so, insert an unknown word token
                    if j >= len(data_by_seq[i]) or k >= len(data_by_seq[i][j]):
                        wid_datum = np.array([self.vocab_w2i[UNKNOWN_WORD]])

                    else:
                        line_of_code = data_by_seq[i][j][k]
                        wid_datum = np.array(self.loc_id_to_coded[str(line_of_code.file_id) +
                                                                  "#" + str(line_of_code.line_number)])
                        labels[i][j] = data_by_seq[i][j][0].label
                        label_weights[i][j] = 1.0

                    data[i][j][k][0:wid_datum.shape[0]] = wid_datum
                    data_wts[i][j][k][0:wid_datum.shape[0]] = np.ones(wid_datum.shape[0])

        return torch.as_tensor(data), torch.as_tensor(data_wts), torch.as_tensor(labels), torch.as_tensor(label_weights)
