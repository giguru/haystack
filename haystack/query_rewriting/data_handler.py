import os
import logging

import numpy as np
import json
# Use external dependency Spacy, because QuReTec also uses Spacy
import spacy

from farm.data_handler.processor import Processor
from farm.data_handler.samples import Sample
from farm.data_handler.utils import pad
from farm.modeling.tokenization import truncate_sequences
from spacy.tokens import Token
from transformers import BertTokenizer
from typing import List
import re

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

cached_parsed_spacy_token = {}


def get_spacy_parsed_word(word: str) -> Token:
    if word not in cached_parsed_spacy_token:
        cached_parsed_spacy_token[word] = nlp(word)[0]
    return cached_parsed_spacy_token[word]


base = os.path.dirname(os.path.abspath(__file__))


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
    Used to handle the QuAC that come in json format.
    For more details on the dataset format, please visit: https://huggingface.co/datasets/quac
    For more information on using custom datasets, please visit: https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb
    """

    def __init__(self,
                 tokenizer: BertTokenizer,
                 max_seq_len=300,
                 train_split: int = None,
                 test_split: int = None,
                 dev_split: int = None,
                 verbose: bool = True,
                 data_dir=base + '/canard/voskarides_preprocessed',
                 train_filename: str = 'train_gold_supervision.json',
                 test_filename: str = 'test_gold_supervision.json',
                 dev_filename: str = 'dev_gold_supervision.json',
                 ):
        """
        :param: max_seq_len. The original authors of QuReTec have provided 300 as the max sequence length.
        """
        self._verbose = verbose

        # Always log this, so users have a log of the settings of their experiments
        logger.info(f"{self.__class__.__name__} with max_seq_len={max_seq_len}")

        train_filename = data_dir + '/' + train_filename
        test_filename = data_dir + '/' + test_filename
        dev_filename = data_dir + '/' + dev_filename

        self.datasets = {
            'train': json.load(open(train_filename))[0:train_split],
            'test': json.load(open(test_filename))[0:test_split],
            'dev': json.load(open(dev_filename))[0:dev_split]
        }
        super(CanardProcessor, self).__init__(tokenizer=tokenizer,
                                              max_seq_len=max_seq_len,
                                              train_filename=train_filename,
                                              dev_filename=dev_filename,
                                              test_filename=test_filename,
                                              data_dir=data_dir,
                                              dev_split=0,
                                            )

        self.add_task(name="ner",
                      metric="F1",
                      label_name=CanardProcessor.label_name_key,
                      task_type="classification",
                      label_list=self.get_labels()
                      )
        self.label_to_id = {label: i for i, label in enumerate(self.get_labels(), 1)}
        self.id_to_label = {i: label for i, label in enumerate(self.get_labels(), 1)}

    @staticmethod
    def get_labels():
        return [
            CanardProcessor.labels['NOT_RELEVANT'],
            CanardProcessor.labels['RELEVANT'],
            "[CLS]",
            "[SEP]"
        ]

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
        label_mask[0] = 0  # mask

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
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "label_ids": label_ids,
            "valid_ids": valid,
            "label_mask": label_mask,
        }]

