import os
import logging
import numpy as np

# Use external dependency Spacy, because QuReTec also uses Spacy
import spacy

from datasets import load_dataset, ReadInstruction
from farm.data_handler.processor import Processor
from farm.data_handler.samples import Sample
from farm.data_handler.utils import pad
from farm.modeling.tokenization import Tokenizer, truncate_sequences
from transformers import BertTokenizer
from typing import List


logger = logging.getLogger(__name__)


def qurectec_sample_to_features_text(sample: Sample,
                                     max_seq_len: int,
                                     tokenizer: BertTokenizer,
                                     preprocessor) -> list:
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :param max_seq_len: Sequences are truncated after this many tokens
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    """
    inputs = sample.tokenized
    input_ids, token_type_ids, special_tokens_mask, start_of_word = inputs["input_ids"], inputs["token_type_ids"], inputs['special_tokens_mask'], inputs['start_of_word']

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # This method was made for QuReTec by Voskarides and they explicitly mention "We mask out the output of <CLS> and
    # the current turn terms, since we are not interested in predicting a label for those", so mask the history only
    attention_mask = [0] * len(input_ids)
    special_tokens_positions = np.where(np.array(special_tokens_mask) == 1)[0]

    history_start_pos = special_tokens_positions[0]
    history_end_pos = special_tokens_positions[1]
    for pos in range(history_start_pos, history_end_pos):
        # Only put attention on start of words
        if start_of_word[pos]:
            attention_mask[pos] = 1

    # Because of the structure of haystack, you don't have access to the sample data in the training loop. However,
    # the target vector required for computing the loss for QuReTec is not in QuAC. So create the target vector here,
    # so it is available
    parsed_context = preprocessor(sample.clear_text[QuacProcessor.context_key])
    parsed_context_lemmas = [t.lemma for t in parsed_context]
    parsed_history = preprocessor(" ".join(sample.clear_text[QuacProcessor.previous_question_key]))

    # Padding up to the sequence length.
    pad_on_left = False
    token_type_ids = pad(token_type_ids, max_seq_len, 0, pad_on_left=pad_on_left)
    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)
    attention_mask = pad(attention_mask, max_seq_len, 0, pad_on_left=pad_on_left)

    # QuReTec quotation: "We apply lowercase, lemmatization and stop-word removal to qi∗, q1:i−1 and qi using Spacy
    # before calculating term overlap in Equation 2"

    # If a term in the question history is present in the question's context, then the term is relevant
    relevant_tokens_from_history = [(1 if t.is_stop is False and t.lemma in parsed_context_lemmas else 0) for t in parsed_history]
    target = [0] * len(input_ids)
    parsed_hist_iter_idx = 0
    for pos in range(history_start_pos, history_end_pos):
        if start_of_word[pos]:
            if 0 <= parsed_hist_iter_idx < len(relevant_tokens_from_history) and relevant_tokens_from_history[parsed_hist_iter_idx] == 1:
                target[pos] = 1
            elif parsed_hist_iter_idx >= len(relevant_tokens_from_history): # In this case, BERT has split the last word
                KeyError(f"Weird error for sample {sample.id}"
                         f"\nSpacy lemmas: {str([token.lemma_ for token in parsed_history])}"
                         f"\nRelevant tokens: {str(relevant_tokens_from_history)}"
                         f"\nBert tokenized text: {str(sample.clear_text['tokenized_text'])}"
                         f"\nBert tokens: {str(tokenizer.convert_ids_to_tokens(input_ids))}"
                         )

            parsed_hist_iter_idx += 1

    assert len(input_ids) == max_seq_len, f"The input_ids vector has length {len(input_ids)}"
    assert len(padding_mask) == max_seq_len, f"The padding_mask vector has length {len(padding_mask)}"
    assert len(token_type_ids) == max_seq_len, f"The token_type_ids vector has length {len(token_type_ids)}"
    assert len(attention_mask) == max_seq_len, f"The attention vector has length {len(attention_mask)}"
    assert len(target) == max_seq_len, f"The target vector has length {len(target)}"

    # Return a list with a features dict
    return [{
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "attention_mask": attention_mask,
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

    prev_word = None
    for word_idx, word in enumerate(tokenized2.encodings[0].tokens):
        if word[:2] == '##'\
                or (word == 's' and prev_word == "'"):  # TODO This should match Spacy's tokenizer
            start_of_word2[word_idx] = 0
        prev_word = word

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
    """
    Used to handle the QuAC that come in json format.
    For more details on the dataset format, please visit: https://huggingface.co/datasets/quac
    For more information on using custom datasets, please visit: https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb
    """

    def __init__(self, tokenizer, max_seq_len):
        # Load dataset from HuggingFace
        cache_dir = '~/datasets'
        datasets = load_dataset('quac',
                                cache_dir=cache_dir,
                                split=[ReadInstruction('train', to=10), ReadInstruction('validation', to=10)],
                                )
        self.datasets = {'train': datasets[0], 'validation': datasets[1]}
        self.train_filename = os.path.basename(self.datasets.get('train').cache_files[0].get('filename'))
        self.test_filename = os.path.basename(self.datasets.get('validation').cache_files[0].get('filename'))
        data_dir = os.path.dirname(self.datasets.get('train').cache_files[0].get('filename'))
        super(QuacProcessor, self).__init__(tokenizer=tokenizer,
                                            max_seq_len=max_seq_len,
                                            train_filename=self.test_filename,
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

    def file_to_dicts(self, file: str) -> [dict]:
        map_func = lambda example: example
        train_filename_path = self.data_dir / self.train_filename
        test_filename_path = self.data_dir / self.test_filename
        if file == test_filename_path:
            return self.datasets.get('validation').map(map_func)
        if file == train_filename_path:
            return self.datasets.get('train').map(map_func)
        raise ValueError(f'Please use the training file {train_filename_path}\n or test file {test_filename_path}')

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        Creates one sample from one dict consisting of the query, positive passages and hard negative passages
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
            if index > 0:
                # BERT model requires the tokenized text to start with a CLS token and end with a SEP token
                # The authors of QuReTec by Voskarides et al. decided to seperate the history and the current question
                # with a SEP token
                tokenized_text = f"{self.tokenizer.cls_token} {' '.join(prev_questions)} {self.tokenizer.sep_token} {question} {self.tokenizer.sep_token}"
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
                                          QuacProcessor.previous_question_key: prev_questions.copy(),
                                          QuacProcessor.context_key: dictionary['context'],
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
        )

