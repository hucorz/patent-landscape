import re
import string
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch


class TextTokenizer:
    punct_regex = re.compile(f"([{string.punctuation}‘’])")
    spaces_regex = re.compile(r"\s{2,}")
    number_regex = re.compile(r"\d+")
    tokenizer = None

    def __init__(self):
        pass

    # def tokenize_to_onehot_matrix(self, text_series, vocab_size, tokenizer=None):
    #     if tokenizer is None:
    #         print(
    #             f"No tokenizer or vocab supplied, so using vocab size ({vocab_size}) and series to build new ones"
    #         )
    #         one_hot_tokenizer = OneHotTokenizer(
    #             num_words=vocab_size,
    #             split=",",
    #             filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
    #             lower=False,
    #         )
    #         one_hot_tokenizer.fit_on_texts(text_series)
    #         one_hot_tokenizer.index_word = {
    #             idx: word for word, idx in one_hot_tokenizer.word_index.items()
    #         }
    #     text_one_hot = one_hot_tokenizer.texts_to_matrix(text_series)
    #     return one_hot_tokenizer, text_one_hot

    def tokenize(
        self,
        text,
        normalize_numbers=True,
        lowercase=True,
        remove_punct=True,
        lemmatize=False,
    ):
        plain_text = text
        if not isinstance(plain_text, str):
            raise Exception(plain_text, type(plain_text))

        preprocessed = plain_text.replace("'", "")
        if lowercase:
            preprocessed = preprocessed.lower()

        if remove_punct:
            preprocessed = self.punct_regex.sub(" ", preprocessed)
        else:
            preprocessed = self.punct_regex.sub(r" \1 ", preprocessed)

        preprocessed = self.spaces_regex.sub(" ", preprocessed)
        if normalize_numbers:
            preprocessed = self.number_regex.sub("_NUMBER_", preprocessed)

        if lemmatize:
            # This requires NLTK or other lemmatizer library
            preprocessed = self.nltk_lemmatize(preprocessed)

        return preprocessed.split()

    def tokenize_series(
        self,
        text_series,
        normalize_numbers=True,
        lowercase=True,
        remove_punct=True,
        lemmatize=False,
    ):
        '''
        text_series: pandas SeriesF
        '''
        return text_series.apply(self.tokenize)

    def nltk_lemmatize(self, text):
        # Implement the lemmatization using nltk or any other library
        # TODO: Implement lemmatization
        raise NotImplementedError


class OneHotTokenizer:
    """
    用来处理 cpc 和 refs 的 tokenizer
    """

    def __init__(
        self,
        num_words=None,
        split=" ",
        filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
    ):
        self.num_words = num_words
        self.split = split
        self.filters = filters
        self.lower = lower
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self.word_index = None

    def _preprocess_text(self, text):
        if self.lower:
            text = text.lower()
        text = text.translate(str.maketrans(self.filters, " " * len(self.filters)))
        return text

    def fit_on_texts(self, texts):
        """
        texts: list of strings, 每个 string 代表的格式： aaa, bbb, ccc
        """
        counter = Counter()
        for line in texts:
            line = self._preprocess_text(line)
            tokens = line.split(self.split)
            counter.update(tokens)
        if self.num_words:
            most_common = counter.most_common(self.num_words)
            vocab_list = [word for word, _ in most_common]
            self.vocab = build_vocab_from_iterator([vocab_list], specials=["<unk>"])
        else:
            self.vocab = build_vocab_from_iterator([counter.keys()], specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.word_index = {word: idx for word, idx in self.vocab.get_stoi().items()}

    def texts_to_matrix(self, texts):
        """
        texts: list of strings, 每个 string 代表的格式： aaa, bbb, ccc
        获得 one-hot 编码矩阵
        """
        if self.vocab is None:
            raise ValueError("The tokenizer has not been fitted on any texts yet.")

        def one_hot_encode(line):
            line = self._preprocess_text(line)
            tokens = line.split(self.split)
            indices = [self.vocab[token] for token in tokens]
            one_hot = torch.zeros(len(self.vocab))
            one_hot[indices] = 1
            return one_hot

        text_one_hot = torch.stack([one_hot_encode(line) for line in texts])
        return text_one_hot
