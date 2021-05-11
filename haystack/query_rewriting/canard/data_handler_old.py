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

BERT_TOKEN_IDS = {
    '?': 1029, ',': 1010
}

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
        source_of_relevance = [t.lemma for t in preprocessor(sample.clear_text[QuacProcessor.context_key])]
    else:
        # Otherwise use gold standard rewrites from the CANARD dataset
        source_of_relevance = [t.lemma for t in preprocessor(sample.clear_text[QuacProcessor.rewrite_key])]

    parsed_prev_questions = preprocessor(" ".join(sample.clear_text[QuacProcessor.previous_question_key]))
    parsed_prev_questions_text = [t.lower_ for t in parsed_prev_questions]
    current_question = [t.lemma for t in preprocessor(sample.clear_text[QuacProcessor.label_name_key])]
    current_question_text = [t.lower_ for t in preprocessor(sample.clear_text[QuacProcessor.label_name_key])]
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
                # Only put attention on start of words, since the paper says "The term classification
                # layer is applied on top of the representation of the first sub-token of each term"
                is_not_punctuation_mark = running_word not in []  # ['?', ',', '-', '.']
                if is_not_punctuation_mark:
                    # Only put focus on punctuation mark
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


class QuacProcessor(Processor):
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
                 train_split = None,
                 validation_split = None,
                 distant_supervision = False,
                 verbose = True,
                 with_rewrites_only = True,
                 keep_initial_questions = False
                 ):
        self._distant_supervision = distant_supervision
        self._verbose = verbose
        self._with_rewrites_only = with_rewrites_only
        self._keep_initial_questions = keep_initial_questions

        # Always log this, so users have a log of the settings of their experiments
        logger.info(f"QuacProcessor with max_seq_len={max_seq_len},"
                    f"distant_supervision={distant_supervision},"
                    f"with_rewrites_only={with_rewrites_only}")

        # Load dataset from HuggingFace
        cache_dir = '~/datasets'
        datasets = load_dataset('quac',
                                cache_dir=cache_dir,
                                split=[
                                    ReadInstruction('train', to=train_split),
                                    ReadInstruction('validation', to=validation_split)
                                ]
                                )
        self.datasets = {
            'train': datasets[0],
            'validation': datasets[1],
        }
        self.train_filename = os.path.basename(self.datasets.get('train').cache_files[0].get('filename'))
        self.test_filename = os.path.basename(self.datasets.get('validation').cache_files[0].get('filename'))
        data_dir = os.path.dirname(self.datasets.get('train').cache_files[0].get('filename'))
        super(QuacProcessor, self).__init__(tokenizer=tokenizer,
                                            max_seq_len=max_seq_len,
                                            train_filename=self.train_filename,
                                            dev_filename=None, # QuAC does not have a dev dataset.
                                            test_filename=self.test_filename,
                                            data_dir=data_dir,
                                            dev_split=0,
                                            )

        self.add_task(name="question_rewriting",
                      metric="F1",
                      label_name=QuacProcessor.label_name_key,
                      task_type="classification",
                      label_list=['target']
                      )

        # Please install this via terminal command "python3 -m spacy download en_core_web_sm"
        self._preprocessor = spacy.load("en_core_web_sm",
                                        disable=["parser", "textcat"]  # see https://spacy.io/usage/processing-pipelines
                                        )
        self._set_canard()


    def _set_canard(self):
        self.canard = {}
        base = os.path.dirname(os.path.abspath(__file__))
        files = {
            'train': json.load(open(base+'/canard/train.json')),
            'test': json.load(open(base+'/canard/test.json'))
        }
        for key, list_of_dicts in files.items():
            dict_per_question_id = defaultdict(dict)
            for index, entry in enumerate(list_of_dicts):
                dialog_id = entry['QuAC_dialog_id']
                question = entry['Question']
                dict_per_question_id[dialog_id][question] = {
                    'rewrite': entry['Rewrite'],
                    'history': entry['History']
                }
            self.canard[key] = dict_per_question_id

    def file_to_dicts(self, file: str) -> [dict]:
        train_filename_path = self.data_dir / self.train_filename
        test_filename_path = self.data_dir / self.test_filename

        # rewritten_question =  if did in self.canard and index < len(self.canard[did]) else None
        if file == test_filename_path:
            dicts = self.datasets.get('validation')
            canard_key = 'test'
        elif file == train_filename_path:
            dicts = self.datasets.get('train')
            canard_key = 'train'
        else:
            raise ValueError(f'Please use the training file {train_filename_path}\n or test file {test_filename_path}')

        if self._with_rewrites_only:
            # QuReTec only uses questions from QuAC with gold rewrites from Canard
            # " When training with gold supervision (gold standard query resolutions), we use the train split
            # from [15], which is a subset of the train split of [7]; all the questions therein have gold
            # standard resolutions."
            filtered_dicts = list(filter(lambda d: d['dialogue_id'] in self.canard[canard_key] and len(d['questions']) > 1, dicts))
            n_queries = sum([len(d['questions']) for d in filtered_dicts]) - len(filtered_dicts)
            if self._verbose:
                logger.info(f"After filtering, there are {len(filtered_dicts)} dialogues with {n_queries} non-first queries. {len(dicts) - len(filtered_dicts)} "
                            f" were excluded from QuAC {os.path.basename(file)}, because they have "
                            f"no rewrites in CANARD")

            return [add_rewrites(d, canard_key=canard_key) for d in filtered_dicts]
        else:
            return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        :param dictionary: The dictionary contains the following keys:
            dialogue_id:            ID of the dialogue.
            wikipedia_page_title:   title of the Wikipedia page.
            background:             first paragraph of the main Wikipedia article.
            section_tile:           Wikipedia section title.
            context:                Wikipedia section text.
            turn_ids:               list of identification of dialogue turns. One list of ids per dialogue.
            questions:              list of questions in the dialogue. One list of questions per dialogue.
            followups:              list of followup actions in the dialogue. One list of followups per dialogue. y: follow, m: maybe follow yp, n: don't follow up.
            yesnos:                 list of yes/no in the dialogue. One list of yes/nos per dialogue. y: yes, n: no, x: neither.
            answers:                dictionary of answers to the questions (validation step of data collection)
                answer_starts:      list of list of starting offsets. For training, list of single element lists (one answer per question).
                texts:              list of list of span texts answering questions. For training, list of single element lists (one answer per question).
            orig_answers:           dictionary of original answers (the ones provided by the teacher in the dialogue)
                answer_starts:      list of starting offsets
                texts:              list of span texts answering questions.
        """
        # Create a sample for each question
        samples = []
        prev_questions = []
        for index, question in enumerate(dictionary['questions']):
            # The idea of the QuacProcessor is using it as a TermClassifier for finding the relevant terms in
            # question history. So there must be a history.
            if self._keep_initial_questions is True or index > 0:
                did = dictionary['dialogue_id']
                if self._with_rewrites_only and question not in self.canard[dictionary['source']][did]:
                    continue
                canard_data = self.canard[dictionary['source']][did][question]
                history = canard_data['history']  # Using canard history, because the

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

                samples.append(Sample(id=dictionary['turn_ids'][index],
                                      clear_text={
                                          QuacProcessor.label_name_key: question,
                                          'tokenized_text': tokenized_text,
                                          QuacProcessor.previous_question_key: history,
                                          QuacProcessor.context_key: dictionary['context'],
                                          QuacProcessor.rewrite_key: canard_data['rewrite'],
                                          # 'answer': dictionary['orig_answers']['texts'][index],
                                          # 'answer_start': dictionary['orig_answers']['answer_starts'][index],
                                          # 'section_title': dictionary['section_title'],
                                          # 'wikipedia_page_title': dictionary['wikipedia_page_title'],
                                      },
                                      tokenized=tokenized)
                               )
            prev_questions.append(question)


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

