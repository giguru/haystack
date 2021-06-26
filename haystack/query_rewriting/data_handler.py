import os
import logging

import datasets
# Use external dependency Spacy, because QuReTec also uses Spacy
import spacy

from farm.data_handler.processor import Processor
from farm.data_handler.samples import Sample
from farm.evaluation.metrics import register_metrics
from spacy.tokens import Token
from transformers import BertTokenizer
from typing import List
import re

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

cached_parsed_spacy_token = {}


def get_entities(seq):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, index).

    Example:
        >>> seq = ['REL', 'REL', 'O', '[SEP]']
        >>> get_entities(seq)
        [('REL', 0), ('REL', 1), ('SEP', 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    return [(label, i) for i, label in enumerate(seq) if label != 'O']


def f1_micro(preds, labels):
    true_entities = set(get_entities(labels))
    pred_entities = set(get_entities(preds))

    correct = len(true_entities & pred_entities)
    pred = len(pred_entities)
    true = len(true_entities)

    micro_precision = correct / pred if pred > 0 else 0
    micro_recall = correct / true if true > 0 else 0

    if micro_precision + micro_recall == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    return {
        'micro_recall': micro_recall * 100,
        'micro_precision': micro_precision * 100,
        'micro_f1': micro_f1 * 100
    }


register_metrics('f1_micro', f1_micro)


def get_spacy_parsed_word(word: str) -> Token:
    if word not in cached_parsed_spacy_token:
        cached_parsed_spacy_token[word] = nlp(word)[0]
    return cached_parsed_spacy_token[word]


class CanardProcessor(Processor):

    label_name_key = "question"
    gold_terms = "gold"

    labels = {
        "NOT_RELEVANT": "O",
        "RELEVANT": "REL",
        "[CLS]": "[CLS]",
        "[SEP]": "[SEP]",
    }
    """
    Used to handle the CANARD that come in json format.
    For more details on the dataset format, please visit: https://huggingface.co/datasets/uva-irlab/canard_quretec
    For more information on using custom datasets, please visit: https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb
    """

    def __init__(self,
                 tokenizer: BertTokenizer,
                 dataset_name: str = 'uva-irlab/canard_quretec',
                 max_seq_len=300,
                 train_split: str = None,
                 test_split: str = None,
                 dev_split: str = None,
                 verbose: bool = True,
                 ):
        """
        :param: max_seq_len. The original authors of QuReTec have provided 300 as the max sequence length.
        """
        self._verbose = verbose

        # Always log this, so users have a log of the settings of their experiments
        logger.info(f"{self.__class__.__name__} with max_seq_len={max_seq_len}")

        loaded_datasets = datasets.load_dataset(
            dataset_name,
            split=[
                f"train[{train_split}]" if train_split else 'train',
                f"test[{test_split}]" if test_split else 'test',
                f"validation[{dev_split}]" if dev_split else 'validation'
            ]
        )
        self.datasets = {
            'train': loaded_datasets[0],
            'test': loaded_datasets[1],
            'dev': loaded_datasets[2],
        }
        data_dir = os.path.dirname(self.datasets['train'].cache_files[0]['filename'])
        train_filename = os.path.basename(self.datasets['train'].cache_files[0]['filename'])
        test_filename = os.path.basename(self.datasets['test'].cache_files[0]['filename'])
        dev_filename = os.path.basename(self.datasets['dev'].cache_files[0]['filename'])

        super(CanardProcessor, self).__init__(tokenizer=tokenizer,
                                              max_seq_len=max_seq_len,
                                              train_filename=train_filename,
                                              dev_filename=dev_filename,
                                              test_filename=test_filename,
                                              data_dir=data_dir,
                                              dev_split=0)
        self.add_task(name="ner",
                      metric="f1_micro",
                      label_list=self.get_labels(),
                      label_name="label"  # The label tensor name without "_ids" at the end
                      )
        self.label_to_id = self.label2id()
        self.id_to_label = self.id2label()

    @staticmethod
    def get_labels():
        return [
            '[PAD]',
            CanardProcessor.labels['NOT_RELEVANT'],
            CanardProcessor.labels['RELEVANT'],
            "[CLS]",
            "[SEP]"
        ]

    @staticmethod
    def label2id():
        return {label: i for i, label in enumerate(CanardProcessor.get_labels())}

    @staticmethod
    def id2label():
        return {i: label for i, label in enumerate(CanardProcessor.get_labels())}

    def pad_token_id():
        return CanardProcessor.get_labels().index('[PAD]')

    def file_to_dicts(self, file: str) -> [dict]:

        test_filename_path = self.data_dir / self.test_filename
        if file == test_filename_path:
            return self.datasets['test']

        train_filename_path = self.data_dir / self.train_filename
        if file == train_filename_path:
            return self.datasets['train']

        dev_filename_path = self.data_dir / self.dev_filename
        if file == dev_filename_path:
            return self.datasets['dev']

        raise ValueError(f'Please use the training file {train_filename_path}\n, test file {test_filename_path} or dev file {dev_filename_path}')

    def relevant_terms(self, history: str, gold_source: str):
        word_list = re.findall(r"[\w']+|[.,!?;]", history)
        gold_list = re.findall(r"[\w']+|[.,!?;]", gold_source)
        gold_lemmas = set([get_spacy_parsed_word(w).lemma_ for w in gold_list])
        label_list = []

        for w in word_list:
            l = get_spacy_parsed_word(w).lemma_
            if l in gold_lemmas:
                label_list.append(CanardProcessor.labels['RELEVANT'])
            else:
                label_list.append(CanardProcessor.labels['NOT_RELEVANT'])

        return word_list, label_list

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        """
        question = dictionary['cur_question'].lower()
        history = dictionary['prev_questions'].lower()
        gold_terms = dictionary['gold_terms']

        # The authors of QuReTec by Voskarides et al. decided to separate the history and the current question
        # with a SEP token
        tokenized_text = f"{history} {self.tokenizer.sep_token} {question}"

        # TODO add ability to use another data set than the preprocessed one of Voskarides
        # word_list, label_list = self.relevant_terms(history=tokenized_text,
        #                                             gold_source=dictionary['answer_text_with_window'])
        word_list = dictionary['bert_ner_overlap'][0]
        label_list = dictionary['bert_ner_overlap'][1]

        if len(word_list) != len(dictionary['bert_ner_overlap'][0]): # TODO remove for final commit
            raise ValueError(f"The word list is parsed differently than Voskarides: {word_list}")

        if label_list != dictionary['bert_ner_overlap'][1]: # TODO remove for final commit
            raise ValueError(f"Label list is constructed differently than Voskarides: {word_list}")

        tokenized = self._quretec_tokenize_with_metadata(words_list=word_list, labellist=label_list)

        if len(tokenized["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {tokenized_text}")

        return [
            Sample(id=f"{dictionary['id']}",
                   clear_text={
                       CanardProcessor.label_name_key: question,
                       'tokenized_text': tokenized_text,
                       CanardProcessor.gold_terms: gold_terms,
                   },
                   tokenized=tokenized)
        ]

    def _quretec_tokenize_with_metadata(self, words_list: List[str], labellist: List[str]):
        """
        :param: text
            The entire text without a initial [CLS] and closing [SEP] token. E.g. "Who are you? [SEP] I am your father"
        """
        tokens, labels, valid, label_mask = [], [], [], []

        if len(words_list) != len(labellist):
            raise ValueError(f"The word list (n={len(words_list)}) should be just as long as label list (n={len(labellist)})")

        for i, word in enumerate(words_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        # truncate lists to match short sequence
        if len(tokens) >= self.max_seq_len - 1:
            # Minus two, because CLS will be prepended and a SEP token will be appended
            tokens = tokens[0:(self.max_seq_len - 2)]
            labels = labels[0:(self.max_seq_len - 2)]
            valid = valid[0:(self.max_seq_len - 2)]
            label_mask = label_mask[0:(self.max_seq_len - 2)]

        return {
            "tokens": tokens,
            "labels": labels,
            "label_mask": label_mask,
            'valid': valid,
        }

    def _sample_to_features(self, sample: Sample) -> List[dict]:
        """
        convert Sample into features for a PyTorch model
        """

        label_mask = sample.tokenized['label_mask']
        labels = sample.tokenized['labels']
        tokens = sample.tokenized['tokens']
        valid = sample.tokenized['valid']

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token

        # Prepend CLS token to the start
        segment_ids = [0]
        label_ids = [self.label_to_id[cls_token]]
        ntokens = [cls_token]

        valid.insert(0, 1)
        label_mask.insert(0, 1)

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(self.label_to_id[labels[i]])
        ntokens.append(sep_token)
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(self.label_to_id[sep_token])
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)

        input_mask = [1] * len(input_ids)

        # mask out labels for current turn.
        cur_turn_index = label_ids.index(self.label_to_id[sep_token])

        label_mask = [1] * cur_turn_index + [0] * (len(label_ids) - cur_turn_index)

        # Pad the features
        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        assert len(label_ids) == len(label_mask), f"label_ids has a different length than label_mask. label_ids={label_ids}, label_mask={label_mask}"
        while len(label_ids) < self.max_seq_len:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len
        assert len(valid) == self.max_seq_len
        assert len(label_mask) == self.max_seq_len

        return [{
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "label_ids": label_ids,
            "valid_ids": valid,
            "label_attention_mask": label_mask,
        }]

