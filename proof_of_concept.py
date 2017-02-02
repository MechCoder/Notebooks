import os
import re
from string import punctuation
from collections import Counter


def clean_split(string, choice=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),@!?\'\`]", " ", string)
    string = re.sub(r"(\d+),(\d+)", r"\1\2", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"`", "'", string)
    string = string.replace("\\)", "rrb")
    string = string.replace("\\(", "lrb")
    string = string.replace("''", " ")
    string = string.replace("' ", " ")
    string = re.sub(r" +", " ", string)

    return [s for s in string.split(" ") if s not in punctuation]

pos_dir = "aclImdb/train/pos"
neg_dir = "aclImdb/train/neg"
pos_files = os.listdir(pos_dir)
neg_files = os.listdir(neg_dir)

def build_dict(pos_files, neg_files, pos_dir, neg_dir):
    c = Counter()
    for pos_file in pos_files:
        review = open(os.path.join(pos_dir, pos_file), "r").read()
        words = clean_split(review)
        for word in words:
            c[word] += 1
    for neg_file in neg_files:
        review = open(os.path.join(neg_dir, neg_file), "r").read()
        words = clean_split(review)
        for word in words:
            c[word] += 1
    return c.most_common(50000)

word_dict = build_dict(pos_files, neg_files, pos_dir, neg_dir)
len(word_dict)
