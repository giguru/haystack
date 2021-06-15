import json
import re
import torch
import logging
import numpy as np
import time
import datetime
from pathlib import Path
from typing import List

from farm.train import Trainer
from tqdm import tqdm
from torch import nn
from farm.modeling.tokenization import Tokenizer
from transformers import BertForTokenClassification, BertConfig
from farm.modeling.optimization import initialize_optimizer
from farm.data_handler.data_silo import DataSilo
from haystack import BaseComponent
from haystack.eval import EvalQueryResolution

logger = logging.getLogger(__name__)


def dict_to_string(d: dict):
    regex = '#(\W|\.)+#'
    params_strings = []
    for k,v in d.items():
        params_strings.append(f"{re.sub(regex, '_', str(k))}_{re.sub(regex, '_', str(v))}")
    return "_".join(params_strings)


class QueryResolutionModel(BertForTokenClassification):
    model_binary_file_name = "query_resolution"

    def __init__(self, config: BertConfig, bert_model: str, max_seq_len: int, device):
        super(QueryResolutionModel, self).__init__(config)
        self.bert_model = bert_model
        self.max_seq_len = max_seq_len
        self.config = config
        self._device = device
        logger.info(f"QueryResolution model {config}")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        valid_output = self.transfer_valid_output(sequence_output=sequence_output, valid_ids=valid_ids)
        sequence_output = self.dropout(valid_output).to(self._device)
        logits = self.classifier(sequence_output).to(self._device)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_logits = self.apply_label_attention_mask_classifier_output(attention_mask_label, logits)
                active_labels = self.apply_attention_mask(attention_mask_label, labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

    def transfer_valid_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        return valid_output

    def apply_label_attention_mask_classifier_output(self, attention_mask_label, logits):
        active_loss = attention_mask_label.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        return active_logits

    def apply_attention_mask(self, attention_mask_label, input):
        active_loss = attention_mask_label.view(-1) == 1
        return input.view(-1)[active_loss]


    def save(self, save_dir: str, id_to_label: dict):
        """
        Save the model state_dict and its config file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        logger.info("Saving model to folder: "+str(save_dir))
        model_to_save = self.module if hasattr(self, 'module') else self  # Only save the model it-self
        model_to_save.save_pretrained(Path(save_dir))
        self._save_config(save_dir, id_to_label=id_to_label)

    def _save_config(self, save_dir: str, id_to_label: dict):
        """
        Saves the config as a json file.
        :param save_dir: Path to save config to
        """
        model_config = self.config.to_dict()
        model_config['bert_model'] = self.bert_model
        model_config['max_seq_len'] = self.max_seq_len
        model_config['id2label'] = id_to_label
        output_config_file = Path(save_dir) / f"config.json"
        with open(output_config_file, "w") as file:
            json.dump(model_config, file)


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities(seq):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, index).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['REL', 'REL', 'O', '[SEP]']
        >>> get_entities(seq)
        [('REL', 0), ('REL', 1), ('SEP', 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    return [(label, i) for i, label in enumerate(seq) if label != 'O']

class QueryResolution(BaseComponent):
    def __init__(self,
                 config,
                 bert_model: str = 'bert-large-uncased',
                 use_gpu: bool = True,
                 progress_bar: bool = True,
                 tokenizer_args: dict = {},
                 model_args: dict = {},
        ):
        """
        Query resolution for Session based pipeline runs. This component is based on the paper:
        Query Resolution for Conversational Search with Limited Supervision
        """
        # Directly store some arguments as instance properties
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar
        self.sigmoid = nn.Sigmoid()

        # Set derived instance properties
        self.tokenizer = Tokenizer.load(bert_model, **tokenizer_args)
        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        logger.info(f"Using {device}")
        self.model = QueryResolutionModel(config=config, device=device, **model_args)
        self.device = torch.device(device)


    def train(self,
              processor,
              eval_metrics: EvalQueryResolution,
              eval_data_set: str,
              learning_rate: float = 5e-5,
              batch_size: int = 2,
              gradient_clipping: float = 1.0,
              n_gpu: int = 1,
              optimizer_name: str = 'AdamW',
              evaluate_every: int = 200,
              print_every: int = 200,
              epsilon: float = 1e-8,
              n_epochs: int = 3,
              save_dir: str = "saved_models",
              disable_tqdm: bool = False,
              grad_acc_steps: int = 2,
              weight_decay: float = 0.01,
              datasilo_args: dict = None,
              num_warmup_steps: int = 200,
              early_stopping: int = None,
              ):
        """
        :param: n_epochs
            Voskarides et al. use 3 epochs
        """
        logger.info(f'Training QueryResolution with batch_size={batch_size}, gradient_clipping={gradient_clipping}, '
                    f'epsilon={epsilon}, n_gpu={n_gpu}, grad_acc_steps={grad_acc_steps}, evaluate_every={evaluate_every}, '
                    f'print_every={print_every}, early_stopping={early_stopping}')
        if datasilo_args is None:
            datasilo_args = {
                "caching": False,
            }

        self.data_silo = DataSilo(processor=processor,
                                  batch_size=batch_size,
                                  distributed=False,
                                  max_processes=1,
                                  **datasilo_args
                                  )

        # Create an optimizer
        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={"name": optimizer_name,
                            "correct_bias": True,
                            "weight_decay": weight_decay,
                            "eps": epsilon
                            },
            schedule_opts={"name": "LinearWarmup",
                           "num_warmup_steps": num_warmup_steps,
                           },
            n_batches=len(self.data_silo.loaders["train"]),
            grad_acc_steps=grad_acc_steps,
            n_epochs=n_epochs,
            device=self.device,
        )

        params_dict = {
            'learning_rate': learning_rate,
            'eps': epsilon,
            'weight_decay': weight_decay,
            'hidden_dropout_prob': self.model.config.hidden_dropout_prob
        }
        model_save_dir = Path(save_dir) / ("query_resolution_" + dict_to_string(params_dict))

        # Set in training mode
        self.model.train()

        loss = 0
        total_loss = 0

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            data_silo=self.data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=self.device,
        )
        trainer.train()
        """
        # TODO use trainer
        for epoch in range(n_epochs):
            t0 = time.time()
            train_data_loader = self.data_silo.get_data_loader("train")

            progress_bar = tqdm(train_data_loader, disable=disable_tqdm)
            for step, batch in enumerate(progress_bar):
                torch.cuda.empty_cache()
                # Calculate elapsed time in minutes.
                elapsed = datetime.timedelta(seconds=int(round(time.time() - t0)))
                progress_bar.set_description(f"Train epoch {epoch + 1}/{n_epochs} (Cur. train loss: {loss:.4f}, Total loss: {total_loss:.4f}) Elapsed {elapsed}")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}
                loss, logits = self.model(input_ids=batch['input_ids'],
                                          token_type_ids=batch['segment_ids'],
                                          attention_mask=batch['input_mask'],
                                          labels=batch['label_ids'],
                                          valid_ids=batch['valid_ids'],
                                          attention_mask_label=batch['label_mask'])
                loss = loss.mean()
                loss.backward()
                loss_val = loss.item()
                total_loss += loss_val

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)

                optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # TODO is this necessary when using model.zero_grad()?
                lr_schedule.step()  # Update the learning rate
                # Always clear previous gradients before performing backward pass. PyTorch doesn't do this automatically
                # accumulating the gradients is "convenient while training RNNs"
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()
                label_list = processor.get_labels()
                if print_every and step % print_every == 0:
                    predicted_tokens, gold_tokens = self._get_results(batch=batch, logits=logits, label_list=label_list)
                    logger.info(f"The items used for metrics are: \n"
                                f"Predicted words: {predicted_tokens}\n"
                                f"Gold tokens: {gold_tokens}")

                if evaluate_every and step % evaluate_every == 0:
                    logger.info(f"Evaluating at step {step}")
                    self.eval(data_loader=self.data_silo.get_data_loader(eval_data_set),
                              metrics=eval_metrics,
                              label_list=label_list)

                    # Eval sets the model in eval mode, so set it to training mode again
                    self.model.train()

                    if eval_metrics.last_recorded_is_the_highest_metric():
                        checkpoint_model_save_dir = str(model_save_dir) + ".ckpt-" + str(step)
                        self.model.save(checkpoint_model_save_dir, id_to_label=processor.id_to_label)
                        self.tokenizer.save_pretrained(save_directory=checkpoint_model_save_dir)

                if early_stopping and step >= early_stopping:
                    break
        """
        # TODO save the best performing model
        self.model.save(model_save_dir, id_to_label=processor.id_to_label)
        self.tokenizer.save_pretrained(save_directory=str(model_save_dir))

    def _get_results(self, batch, logits, label_list):
        """
        :param: logits
            Two dimensional logits. N x M where N is the number of samples in a batch and M is the tensor length of one
            sample.
        """
        logits = torch.argmax(nn.functional.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_to_id_map = {label: i for i, label in enumerate(label_list, 1)}
        y_true, y_pred, terms = [], [], []
        label_ids = batch['label_ids'].to('cpu').numpy()
        input_ids = batch['input_ids'].to('cpu').numpy()
        valid_ids = batch['valid_ids'].to('cpu').numpy()
        for i, label in enumerate(label_ids):
            temp_1, temp_2, temp_3 = [], [], []

            for j, m in enumerate(label):
                if j == 0:  # skip initial CLS
                    continue
                elif label_ids[i][j] == label_to_id_map['[SEP]']:  # break at the first [SEP] token
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    tmp = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    x_input_tokens = []
                    for jj in range(1, len(tmp)):  # skip initial CLS
                        token = tmp[jj]
                        if token == '[PAD]':
                            break
                        if valid_ids[i][jj] == 1:
                            x_input_tokens.append(token)
                        else:
                            x_input_tokens[-1] += token

                    # remove bert tokenization chars ## from tokens
                    terms.append([s.replace('##', '') for s in x_input_tokens])
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map.get(logits[i][j], 'O'))
                    temp_3.append(input_ids[i][j])

        return y_pred, y_true

    def _print_classifier_result(self,
                                 input_token_ids: torch.Tensor,
                                 target_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 verbose: bool = True):
        relevant_token_ids = {'output': [], 'target': [], 'input': []}

        # QuReTec is a binary classifier. The paper does say it uses a sigmoid layer, but it does not say how to get to
        # an output of zeros and ones. Rounding makes sense, i.e. splitting on 0.5
        rounded_output = output_tensor.float().round()
        for idx, token_id in enumerate(input_token_ids):
            token_id_int = int(token_id)
            if attention_mask[idx] == 1.0:
                relevant_token_ids['input'].append(token_id_int)
            if target_tensor[idx] == 1.0:
                relevant_token_ids['target'].append(token_id_int)
            if rounded_output[idx] == 1.0:
                relevant_token_ids['output'].append(token_id_int)

        if verbose:
            relevant_tokens = {k: self.tokenizer.convert_ids_to_tokens(ids=relevant_token_ids[k]) for k in relevant_token_ids}
            logger.info(f'\nGiven input: {" ".join(self.tokenizer.convert_ids_to_tokens(ids=input_token_ids)).replace(" [PAD]", "")}\n'
                        f'and attention: {str(self.tokenizer.convert_ids_to_tokens(ids=relevant_token_ids["input"]))}\n'
                        f'and target     {str(relevant_tokens["target"])}\n'
                        f'the result is: {str(relevant_tokens["output"])}')

    def _get_predicted_and_gold_tokens(self, terms: List, predicted_labels: List, gold_labels: List):
        """
        Args:
            gold_labels : 2d array. Ground truth (correct) target values.
            predicted_labels : 2d array. Estimated targets as returned by a tagger.
        """
        # Paper says: "We apply lowercase, lemmatization and stopword removal to qi∗,
        # q1:i−1 and qi using Spacy12 before calculating term overlap in Equation 2.", so
        # lemmatise the full word and remove non-stopwords.
        true_entities = set(get_entities(gold_labels))
        pred_entities = set(get_entities(predicted_labels))

        try:
            predicted_terms = [terms[tup[1]] for tup in pred_entities]
            gold_terms = [terms[tup[1]] for tup in true_entities]
            return predicted_terms, gold_terms
        except IndexError as e:
            print(f"{e} with terms={terms} \nand true_entities={true_entities} \nand pred_entities={pred_entities}")
            raise e


    def eval(self,
             data_loader,
             label_list,
             metrics: EvalQueryResolution,
             disable_tqdm: bool = False):
        self.model.to(self.device)
        self.model.eval()

        t0 = time.time()

        progress_bar = tqdm(data_loader, disable=disable_tqdm)

        for step, batch in enumerate(progress_bar):
            # Calculate elapsed time in minutes.
            elapsed = datetime.timedelta(seconds=int(round(time.time() - t0)))
            progress_bar.set_description(f"Evaluating in progress. Elapsed: {elapsed}")

            # Move batch of samples to device
            batch = {key: batch[key].to(self.device) for key in batch}
            with torch.no_grad():
                logits = self.model(input_ids=batch['input_ids'],
                                    token_type_ids=batch['segment_ids'],
                                    attention_mask=batch['input_mask'],
                                    valid_ids=batch['valid_ids'],
                                    attention_mask_label=batch['label_mask'])
            predicted_tokens, gold_tokens = self._get_results(batch=batch, logits=logits, label_list=label_list)
            for i in range(len(predicted_tokens)):
                metrics.run(predicted_tokens=predicted_tokens[i], gold_tokens=gold_tokens[i])
        progress_bar.close()
        logger.info("Printing metrics")
        metrics.print(mode="query_resolution", use_logged_values=False)
        metrics.add_to_log()
        return metrics

    def resolve(self, question: str, history):
        """
        Rewrite a single question + history ad hoc instead of using a data set.
        """
        # TODO
        pass