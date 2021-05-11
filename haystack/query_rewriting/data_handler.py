import os
import logging
from collections import defaultdict

import numpy as np
import json
# Use external dependency Spacy, because QuReTec also uses Spacy
import spacy

from datasets import load_dataset, ReadInstruction
from farm.data_handler.processor import Processor
from farm.data_handler.samples import Sample
from farm.data_handler.utils import pad
from farm.modeling.tokenization import truncate_sequences
from transformers import BertTokenizer
from typing import List


logger = logging.getLogger(__name__)


def add_rewrites(d, canard_key):
    d.update({'source': canard_key})
    return d

def qurectec_sample_to_features_text(sample: Sample,
                                     max_seq_len: int,
                                     tokenizer: BertTokenizer,
                                     preprocessor,
                                     distant_supervision: bool,
                                     pad_on_left = False,
                                     debugging = True
                                     ) -> List[dict]:
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :param max_seq_len: Sequences are truncated after this many tokens
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    """
    inputs = sample.tokenized
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    special_tokens_positions = np.where(np.array(inputs['special_tokens_mask']) == 1)[0]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # This method was made for QuReTec by Voskarides and they explicitly mention "We mask out the output of <CLS> and
    # the current turn terms, since we are not interested in predicting a label for those", so mask the history only
    attention_mask = [0] * len(input_ids)

    # Because of the structure of haystack, you don't have access to the sample data in the training loop. However,
    # the target vector required for computing the loss for QuReTec is not in QuAC. So create the target vector
    # here, so it is available during the training loop
    target = [0] * len(input_ids)

    # Bert tokenizers start of words are different from Spacy, but we need the one from Spacy
    start_of_word = [0] * len(input_ids)

    if distant_supervision:
        source_of_relevance = [t.lemma for t in preprocessor(sample.clear_text[CanardProcessor.context_key])]
    else:
        # Otherwise use gold standard rewrites from the CANARD dataset
        source_of_relevance = [t.lemma for t in preprocessor(sample.clear_text[CanardProcessor.rewrite_key])]

    parsed_prev_questions = preprocessor(" ".join(sample.clear_text[CanardProcessor.previous_question_key]))
    parsed_prev_questions_text = [t.lower_ for t in parsed_prev_questions]
    current_question = [t.lemma for t in preprocessor(sample.clear_text[CanardProcessor.label_name_key])]
    current_question_text = [t.lower_ for t in preprocessor(sample.clear_text[CanardProcessor.label_name_key])]
    potentials_words = []
    for t in parsed_prev_questions:
        # QuReTec quotation: "We apply lowercase, lemmatization and stop-word removal to qi∗, q1:i−1 and qi using Spacy
        # before calculating term overlap in Equation 2"

        # If a term in the question history is present in the question context/history and not in the current question,
        # then the term is relevant
        if t.is_stop is False and t.lemma in source_of_relevance and t.lemma not in current_question:
            potentials_words.append(t.lower_)

    target_tokens, not_target_tokens = [], []  # for during debugging

    # The history starts after the initial [CLS] and ends at the second special token, which is [SEP]
    end_for_attention_mask, end_of_input = special_tokens_positions[1], len(input_ids)
    bert_pos, running_spacy_idx = 0, 0
    try:
        while bert_pos < end_of_input:
            if bert_pos in special_tokens_positions:
                running_spacy_idx = 0
                bert_pos += 1
                continue

            start_of_word[bert_pos] = 1
            token_string = tokenizer.convert_ids_to_tokens(input_ids[bert_pos])
            running_word = token_string

            extra = 0
            spacy_word = parsed_prev_questions_text[running_spacy_idx] if bert_pos <= end_for_attention_mask else current_question_text[running_spacy_idx]
            # Prevent looping past the end of the sentence or over special tokens positions
            while len(running_word) < len(spacy_word) and bert_pos + extra < end_of_input and bert_pos + extra + 1 not in special_tokens_positions:
                extra += 1
                running_word += tokenizer.convert_ids_to_tokens(input_ids[bert_pos + extra]).replace("##", "")

            if bert_pos < end_for_attention_mask:

                is_not_punctuation_mark = running_word not in ['?', ',', '-', '.']
                if is_not_punctuation_mark:
                    # Only put attention on start of words, since the paper says "The term classification
                    # layer is applied on top of the representation of the first sub-token of each term"
                    attention_mask[bert_pos] = 1

                if running_word in potentials_words and is_not_punctuation_mark:
                    target[bert_pos] = 1
                    if debugging:
                        target_tokens.append(token_string)
                elif debugging:
                    not_target_tokens.append(token_string)
            running_spacy_idx += 1
            bert_pos += 1 + extra
    except IndexError as e:
        print(e)
    del bert_pos, running_spacy_idx, end_for_attention_mask, end_of_input

    # Padding up to the sequence length.
    token_type_ids = pad(token_type_ids, max_seq_len, 0, pad_on_left=pad_on_left)
    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)
    attention_mask = pad(attention_mask, max_seq_len, 0, pad_on_left=pad_on_left)
    target = pad(target, max_seq_len, 0, pad_on_left=pad_on_left)
    start_of_word = pad(start_of_word, max_seq_len, 0, pad_on_left=pad_on_left)

    assert len(input_ids) == max_seq_len, f"The input_ids vector has length {len(input_ids)}"
    assert len(padding_mask) == max_seq_len, f"The padding_mask vector has length {len(padding_mask)}"
    assert len(token_type_ids) == max_seq_len, f"The token_type_ids vector has length {len(token_type_ids)}"
    assert len(attention_mask) == max_seq_len, f"The attention vector has length {len(attention_mask)}"
    assert len(start_of_word) == max_seq_len, f"The start_of_word vector has length {len(start_of_word)}"
    assert len(target) == max_seq_len, f"The target vector has length {len(target)}"

    # Return a list with a features dict
    return [{
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "attention_mask": attention_mask,
        "start_of_words": start_of_word,
        "token_type_ids": token_type_ids,
        "target": target
    }]


def quretec_tokenize_with_metadata(text_including_special_tokens: str, max_seq_len, tokenizer: BertTokenizer) -> dict:
    tokenized2 = tokenizer(text=text_including_special_tokens,
                           truncation=True,
                           truncation_strategy="longest_first",
                           max_length=max_seq_len,
                           return_token_type_ids=True,
                           return_offsets_mapping=True,
                           return_special_tokens_mask=True,
                           add_special_tokens=False  # The provided text already has special tokens
                           )

    tokens2 = tokenized2["input_ids"]
    offsets2 = np.array([x[0] for x in tokenized2["offset_mapping"]])
    words = np.array(tokenized2.encodings[0].words)

    # TODO check for validity for all tokenizer and special token types
    words[0] = -1
    words[-1] = words[-2]
    words += 1
    start_of_word2 = [0] + list(np.ediff1d(words))

    # The bert tokenizer does not find special tokens that are in the text to be tokenized, so overwrite special tokens
    # mask manually
    special_tokens_mask = tokenizer.get_special_tokens_mask(tokenized2['input_ids'], already_has_special_tokens=True)
    return {"tokens": tokens2,
            "offsets": offsets2,
            "start_of_word": start_of_word2,
            "input_ids": tokenized2['input_ids'],
            'token_type_ids': tokenized2['token_type_ids'],
            'special_tokens_mask': special_tokens_mask,
            }


class CanardProcessor(Processor):
    label_name_key = "question"
    previous_question_key = "previous_questions"
    context_key = "context"
    rewrite_key = "rewrite"
    """
    Used to handle the QuAC that come in json format.
    For more details on the dataset format, please visit: https://huggingface.co/datasets/quac
    For more information on using custom datasets, please visit: https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb
    """

    def __init__(self,
                 tokenizer,
                 max_seq_len=512,  # 512, because that is the maximum number of tokens BERT uses
                 train_split: int = None,
                 test_split: int = None,
                 dev_split: int = None,
                 distant_supervision = False,
                 use_first_questions = False,
                 verbose = True
                 ):
        self._distant_supervision = distant_supervision
        self._verbose = verbose

        # Always log this, so users have a log of the settings of their experiments
        logger.info(f"{self.__class__.__name__} with max_seq_len={max_seq_len},"
                    f"distant_supervision={distant_supervision}, use_first_questions={use_first_questions}"
                    )

        base = os.path.dirname(os.path.abspath(__file__))
        data_dir = base+'/canard'
        train_filename = data_dir + '/train.json'
        test_filename = data_dir + '/test.json'
        dev_filename = data_dir + '/dev.json'
        self.datasets = {
            'train': json.load(open(train_filename))[0:train_split],
            'test': json.load(open(test_filename))[0:test_split],
            'dev': json.load(open(dev_filename))[0:dev_split]
        }
        for key in self.datasets:
            self.datasets[key] = [d for d in self.datasets[key] if (use_first_questions or d['Question_no'] > 1)]

        super(CanardProcessor, self).__init__(tokenizer=tokenizer,
                                              max_seq_len=max_seq_len,
                                              train_filename=train_filename,
                                              dev_filename=dev_filename,
                                              test_filename=test_filename,
                                              data_dir=data_dir,
                                              dev_split=0,
                                            )

        self.add_task(name="question_rewriting",
                      metric="F1",
                      label_name=CanardProcessor.label_name_key,
                      task_type="classification",
                      label_list=['target']
                      )

        # Please install this via terminal command "python3 -m spacy download en_core_web_sm"
        self._preprocessor = spacy.load("en_core_web_sm",
                                        disable=["parser", "textcat"]  # see https://spacy.io/usage/processing-pipelines
                                        )


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

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        :param dictionary: The dictionary contains the following keys:
            History
            QuAC_dialog_id
            Question
            Question_no
            Rewrite
        """
        # Create a sample for each question
        samples = []

        question = dictionary['Question']
        history = dictionary['History']

        # BERT model requires the tokenized text to start with a CLS token and end with a SEP token
        # The authors of QuReTec by Voskarides et al. decided to seperate the history and the current question
        # with a SEP token
        tokenized_text = f"{self.tokenizer.cls_token} {' '.join(history)} {self.tokenizer.sep_token} {question} {self.tokenizer.sep_token}"
        tokenized = quretec_tokenize_with_metadata(text_including_special_tokens=tokenized_text,
                                                   max_seq_len=self.max_seq_len,
                                                   tokenizer=self.tokenizer)

        if len(tokenized["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {tokenized_text}")

        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model.
        # Source: https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name],
                                                           seq_b=None,
                                                           tokenizer=self.tokenizer,
                                                           max_seq_len=self.max_seq_len,
                                                           truncation_strategy='longest_first',
                                                           with_special_tokens=False,  # False, because it already contains special tokens
                                                           stride=0)

        samples.append(Sample(id=f"{dictionary['QuAC_dialog_id']}#q{dictionary['Question_no']}",
                              clear_text={
                                  CanardProcessor.label_name_key: question,
                                  'tokenized_text': tokenized_text,
                                  CanardProcessor.previous_question_key: history,
                                  CanardProcessor.context_key: '',  # TODO get context from quac from QUAC
                                  CanardProcessor.rewrite_key: dictionary['Rewrite'],
                                  # 'answer': dictionary['orig_answers']['texts'][index],
                                  # 'answer_start': dictionary['orig_answers']['answer_starts'][index],
                                  # 'section_title': dictionary['section_title'],
                                  # 'wikipedia_page_title': dictionary['wikipedia_page_title'],
                              },
                              tokenized=tokenized)
                       )
        return samples

    def _sample_to_features(self, sample: Sample) -> List[dict]:
        """
        convert Sample into features for a PyTorch model
        """
        return qurectec_sample_to_features_text(
            sample=sample,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
            preprocessor=self._preprocessor,
            distant_supervision=self._distant_supervision
        )

