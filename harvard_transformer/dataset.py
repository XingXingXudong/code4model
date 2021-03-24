import json
import numpy as np
import pandas as pd
import jieba
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""
    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {
            idx: token
            for token, idx in self._token_to_idx.items()
        }

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.
        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary
        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token
        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index
        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self,
                 token_to_idx=None,
                 unk_token="<UNK>",
                 mask_token="<MASK>",
                 begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({
            'unk_token': self._unk_token,
            'mask_token': self._mask_token,
            'begin_seq_token': self._begin_seq_token,
            'end_seq_token': self._end_seq_token
        })
        return contents

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.
        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class NMTVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, source_vocab, target_vocab, max_source_length,
                 max_target_length):
        """
        Args:
            source_vocab (SequenceVocabulary): maps source words to integers
            target_vocab (SequenceVocabulary): maps target words to integers
            max_source_length (int): the longest sequence in the source dataset
            max_target_length (int): the longest sequence in the target dataset
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """Vectorize the provided indices
        Args:
            indices (list): a list of integers that represent a sequence
            vector_length (int): an argument for forcing the length of index vector
            mask_index (int): the mask_index to use; almost always 0
        """
        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _get_source_indices(self, text):
        """Return the vectorized source text
        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            indices (list): list of integers representing the text
        """
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(
            self.source_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)
        return indices

    def _get_target_indices(self, text):
        """Return the vectorized source text
        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            a tuple: (x_indices, y_indices)
                x_indices (list): list of integers representing the observations in target decoder
                y_indices (list): list of integers representing predictions in target decoder
        """
        indices = [
            self.target_vocab.lookup_token(token) for token in text.split(" ")
        ]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices

    def vectorize(self,
                  source_text,
                  target_text,
                  use_dataset_max_lengths=True):
        """Return the vectorized source and target text
        The vetorized source text is just the a single vector.
        The vectorized target text is split into two vectors in a similar style to
            the surname modeling in Chapter 7.
        At each timestep, the first vector is the observation and the second vector is the target.
        Args:
            source_text (str): text from the source language
            target_text (str): text from the target language
            use_dataset_max_lengths (bool): whether to use the global max vector lengths
        Returns:
            The vectorized data point as a dictionary with the keys:
                source_vector, target_x_vector, target_y_vector, source_length
        """
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(
            source_indices,
            vector_length=source_vector_length,
            mask_index=self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(
            target_text)
        target_x_vector = self._vectorize(
            target_x_indices,
            vector_length=target_vector_length,
            mask_index=self.target_vocab.mask_index)
        target_y_vector = self._vectorize(
            target_y_indices,
            vector_length=target_vector_length,
            mask_index=self.target_vocab.mask_index)
        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": len(source_indices)
        }

    @classmethod
    def from_dataframe(cls, bitext_df):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            bitext_df (pandas.DataFrame): the parallel text dataset
        Returns:
            an instance of the NMTVectorizer
        """
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length = 0
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length,
                   max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(
            contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(
            contents["target_vocab"])

        return cls(source_vocab=source_vocab,
                   target_vocab=target_vocab,
                   max_source_length=contents["max_source_length"],
                   max_target_length=contents["max_target_length"])

    def to_serializable(self):
        return {
            "source_vocab": self.source_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length
        }


class NMTDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        """
        Args:
            surname_df (pandas.DataFrame): the dataset
            vectorizer (SurnameVectorizer): vectorizer instatiated from dataset
        """
        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.validation_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        """Load dataset and make a new vectorizer from scratch
        Args:
            surname_csv (str): location of the dataset
        Returns:
            an instance of SurnameDataset
        """
        text_df = pd.read_csv(dataset_csv, sep='\t')
        train_subset = text_df[text_df.split == 'train']
        return cls(text_df, NMTVectorizer.from_dataframe(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv,
                                         vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use
        Args:
            surname_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of SurnameDataset
        """
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of SurnameVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language,
                                                 row.target_language)

        return {
            "x_source": vector_dict["source_vector"],
            "x_target": vector_dict["target_x_vector"],
            "y_target": vector_dict["target_y_vector"],
            "x_source_length": vector_dict["source_length"]
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def generate_nmt_batches(dataset,
                         batch_size,
                         shuffle=True,
                         drop_last=True,
                         make_mask=True,
                         device="cpu"):
    """A generator function which wraps the PyTorch DataLoader.  The NMT Version """
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(
                device)
        out_data_dict['ntokens'] = (
            out_data_dict['y_target'] != 
            dataset._vectorizer.target_vocab.lookup_token('<MASK>')).data.sum()
        if make_mask:
            out_data_dict['src_mask'] = (
                out_data_dict['x_source'] !=
                dataset._vectorizer.source_vocab.lookup_token('<MASK>')
            ).unsqueeze(-2).to(device)
            tgt_mask = (out_data_dict['x_target'] !=
                        dataset._vectorizer.target_vocab.lookup_token('<MASK>')
                        ).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                subsequent_mask(out_data_dict['x_target'].size(-1)).type_as(
                    tgt_mask.data))
            out_data_dict['tgt_mask'] = tgt_mask.to(device)
        yield out_data_dict


def vec2string(dataset, datadict):
    strdict = {}
    for n in ['x_source', 'x_target', 'y_target']:
        _, st = n.split('_')
        if st == 'source':
            vocab = dataset._vectorizer.source_vocab
        else:
            vocab = dataset._vectorizer.target_vocab
        array = datadict[n].numpy()
        strings = []
        for i in range(array.shape[0]):
            strings.append([vocab.lookup_index(id) for id in array[i, :]])
        strdict[n] = strings
    return strdict

# 处理中-英翻译数据集
def en_token(sq):
    return " ".join(sq.strip().split(' '))


def zh_token(sq):
    return " ".join(jieba.lcut(sq.strip()))


def split(N, tags=('train', 'val', 'test'), ps=(0.7, 0.2, 0.1)):
    return np.random.choice(tags, N, ps)


def data_process(trans_json):
    lines = []
    with open(trans_json, 'r') as fr:
        for i, line in enumerate(fr.readlines()):
            if i % 1001 == 1: 
                print('process: ', i)
            ct = json.loads(line.strip())
            lines.append((en_token(ct['english']), zh_token(ct['chinese'])))
    df = pd.DataFrame(lines, columns=['source_language', 'target_language'])
    df['split'] = split(len(lines))
    return df


def test_data_process():
    trans_json = "/home/liuxd/home/NLP/PyTorchNLPBook/code4model/data/translation2019zh_train.json"
    save_path = "/home/liuxd/home/NLP/PyTorchNLPBook/code4model/data/translation2019zh_train-df.csv"
    rtn = data_process(trans_json)
    rtn.to_csv(save_path, sep='\t', index=False)
    print(rtn)


if __name__ == '__main__':
    dataset = NMTDataset.load_dataset_and_make_vectorizer(
        "/home/liuxd/home/NLP/PyTorchNLPBook/code4model/data/translation2019zh_train-df_200w.csv"
    )
    batchs = generate_nmt_batches(dataset, 2)
    batch = next(batchs)
    print(batch['x_source'])
    print(batch['src_mask'])
    print(batch['x_target'])
    print(batch['tgt_mask'])
    print(batch['y_target'])
    print("==============================")
    print(batch['ntokens'])
    print("==============================")
    print(batch['x_source_length'])
    print(vec2string(dataset, batch))
