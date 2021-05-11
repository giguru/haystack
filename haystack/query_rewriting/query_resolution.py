import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import logging
import numpy as np
from farm.data_handler.utils import is_json

from tqdm import tqdm
import time
from torch import nn
import datetime

from farm.modeling.tokenization import Tokenizer
from transformers import AutoModel
from farm.modeling.optimization import initialize_optimizer
from farm.data_handler.data_silo import DataSilo

from haystack import BaseComponent

logger = logging.getLogger(__name__)


class MicroMetrics:
    def __init__(self):
        self.true_pos, self.false_pos, self.false_neg = (0., 0., 0.)

    def add_to_values(self, predicted_tokens: List[int], gold_tokens: List[int], counts: bool = False):
        predicted_count = {i: predicted_tokens.count(i) if counts else 1 for i in predicted_tokens}
        gold_count = {i: gold_tokens.count(i) if counts else 1 for i in gold_tokens}
        fp, tp, fn = 0, 0, 0

        for i in predicted_count:
            if i not in gold_count:
                fp += predicted_count[i]
            elif predicted_count[i] >= gold_count[i]:
                tp += gold_count[i]
                fp += predicted_count[i] - gold_count[i]
            elif predicted_count[i] < gold_count[i]:
                tp += predicted_count[i]
                fn += gold_count[i] - predicted_count[i]
        # Some id might have never been predicted
        for i in gold_count:
            if i not in predicted_count:
                fn += gold_count[i]

        self.true_pos += tp
        self.false_pos += fp
        self.false_neg += fn

    def calc_metric(self):
        micro_recall = self.true_pos / (self.true_pos + self.false_neg)
        micro_precision = self.true_pos / (self.true_pos + self.false_pos)
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        return {
            'micro_recall': micro_recall,
            'micro_precision': micro_precision,
            'micro_f1': micro_f1
        }


def dict_to_string(d: dict):
    regex = '#(\W|\.)+#'
    params_strings = []
    for k,v in d.items():
        params_strings.append(f"{re.sub(regex, '_', str(k))}_{re.sub(regex, '_', str(v))}")
    return "_".join(params_strings)


class QueryResolutionModel(nn.Module):
    model_binary_file_name = "query_resolution"

    def __init__(self,
                 model_name_or_path: str,
                 linear_layer_size=1024,
                 output_size=512,
                 dropout_prob=0.0
                 ):
        """
        :param linear_layer_size
            1024 is suitable for the bert-large-uncased model from HuggingFace Transformers library
        :param output_size
        """
        super(QueryResolutionModel, self).__init__()
        self.dropout_prob = dropout_prob
        self._linear_layer_size = linear_layer_size
        self.output_size = output_size

        self._bert = AutoModel.from_pretrained(model_name_or_path, hidden_dropout_prob=dropout_prob)
        self._dropout = nn.Dropout(p=dropout_prob)

        self._linear1 = nn.Linear(self._linear_layer_size, output_size)
        logger.info(f"Initiating QueryResolution model with dropout_prod={dropout_prob}")

    def forward(self, **kwargs):
        """
        Watch out! Since the output of this model should be a classification token for each term
        (0 for unrelevant tokens and 1 for relevant tokens). Please make sure you apply the sigmoid function
        on the output of this method to make sure value is between zero and 1.
        This is done, because the QueryResolution pipeline component use a torch.nn.BCELogitsWithLoss which uses
        a sigmoid internally instead of a torch.nn.BCELoss.
        """
        bert_output = self._bert(**kwargs)
        sequence_output = bert_output.last_hidden_state

        # sequence_output has the following shape: (batch_size, sequence_length, output_size)
        # extract the 1st token's embeddings
        dropout_output = self._dropout(sequence_output[:, 0, :].view(-1, self._linear_layer_size))
        linear1_output = self._linear1(dropout_output)
        return linear1_output

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        logger.info("Saving model to folder: "+str(save_dir))
        torch.save(self.state_dict(), f=Path(save_dir) / 'pytorch_model.bin', )
        self._save_config(save_dir)

    def _save_config(self, save_dir: str):
        """
        Saves the config as a json file.
        :param save_dir: Path to save config to
        """
        output_config_file = Path(save_dir) / f"config.json"
        with open(output_config_file, "w") as file:
            json.dump(self._generate_config(), file)

    def _generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = self._bert.config.to_dict()
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if is_json(value) and key[0] != "_":
                config[key] = value
        config["name"] = self.__class__.__name__
        config.pop("config", None)
        return config


class QueryResolution(BaseComponent):
    def __init__(self,
                 model_name_or_path: str = "bert-large-uncased",
                 use_gpu: bool = True,
                 progress_bar: bool = True,
                 tokenizer_args: dict = {},
                 model_args: dict = {}
        ):
        """
        Query resolution for Session based pipeline runs. This component is based on the paper:
        Query Resolution for Conversational Search with Limited Supervision
        """
        # Directly store some arguments as instance properties
        self.model_name_or_path = model_name_or_path
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar
        self.sigmoid = nn.Sigmoid()

        # Set derived instance properties
        self.tokenizer = Tokenizer.load(model_name_or_path, **tokenizer_args)
        self.model = QueryResolutionModel(model_name_or_path=model_name_or_path, **model_args)
        self._prev_eval_results = {'p': 0, 'r': 0, 'f1': 0}

        if use_gpu and torch.cuda.is_available():
            logger.info("Using GPU Cuda")
            self.device = torch.device("cuda")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")

    def train(self,
              processor,
              eval_data_set: str,
              learning_rate: float = 3e-5,
              batch_size: int = 2,
              gradient_clipping: float = 1.0,
              n_gpu: int = 1,
              optimizer_name: str = 'AdamW',
              evaluate_every: int = 200,
              print_every: int = 200,
              epsilon: float = 1e-08,  # TODO QuReTec does not mention epsilon
              n_epochs: int = 1,  # TODO QuReTec paper does not mention epochs.
              save_dir: str = "saved_models",
              disable_tqdm: bool = False,
              grad_acc_steps: int = 2,
              weight_decay: float = 0.01
              ):
        logger.info(f'Training QueryResolution with batch_size={batch_size}, gradient_clipping={gradient_clipping}, '
                    f'epsilon={epsilon}, n_gpu={n_gpu}, grad_acc_steps={grad_acc_steps}')

        self.data_silo = DataSilo(processor=processor,
                                  batch_size=batch_size,
                                  distributed=False,
                                  max_processes=1
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
                           "num_warmup_steps": 0
                           },
            n_batches=len(self.data_silo.loaders["train"]),
            grad_acc_steps=grad_acc_steps,
            n_epochs=n_epochs,
            device=self.device,
        )

        # Set in training mode
        self.model.train()

        loss_fn = nn.BCEWithLogitsLoss()
        loss = 0
        total_loss = 0

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

                # Always clear previous gradients before performing backward pass. PyTorch doesn't do this automatically
                # accumulating the gradients is "convenient while training RNNs"
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     token_type_ids=batch['token_type_ids'])
                target = batch['target'].float()
                loss = loss_fn(outputs, target)
                total_loss += loss
                loss.backward()

                if print_every and step % print_every == 0:
                    # Show an example of the results
                    sigmoid_outputs = self.sigmoid(outputs.detach()).detach()
                    self._get_classifier_result(input_token_ids=batch['input_ids'][0],
                                                attention_mask=batch['attention_mask'][0],
                                                target_tensor=target[0],
                                                output_tensor=sigmoid_outputs[0])
                # Prevent exploding gradients
                if hasattr(optimizer, "clip_grad_norm"):
                    # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                    optimizer.clip_grad_norm(gradient_clipping)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)

                optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # TODO is this necessary when using model.zero_grad()?
                lr_schedule.step()  # Update the learning rate

                if evaluate_every and step % evaluate_every == 0:
                    self.eval(data_loader=self.data_silo.get_data_loader(eval_data_set))

                    # Eval sets the model in eval mode, so set it to training mode again
                    self.model.train()

                # TODO save every x points
                # if checkpoint_every and step % checkpoint_every == 0:

        params_dict = {
            'learning_rate': learning_rate,
            'eps':  epsilon,
            'weight_decay': weight_decay,
            'dropout': self.model.dropout_prob
        }
        model_save_dir = Path(save_dir) / ("query_resolution_" + dict_to_string(params_dict))
        self.model.save(model_save_dir)
        self.tokenizer.save_pretrained(save_directory=str(model_save_dir))

    def _get_classifier_result(self,
                               input_token_ids: torch.Tensor,
                               target_tensor: torch.Tensor,
                               output_tensor: torch.Tensor,
                               attention_mask: torch.Tensor,
                               verbose: bool = True):
        relevant_token_ids = {'output': [], 'target': [], 'input': []}
        # QuReTec is a binary classifier. The paper does say it uses a sigmoid layer, but it does not say how to get to
        # an output of zeros and ones. Rounding makes sense, i.e. splitting on 0.5
        rounded_output = output_tensor.round()
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
            logger.info(f'\nGiven history: {" ".join(self.tokenizer.convert_ids_to_tokens(ids=input_token_ids)).replace(" [PAD]", "")}\n'
                        f'and attention: {str(self.tokenizer.convert_ids_to_tokens(ids=relevant_token_ids["input"]))}\n'
                        f'and target     {str(relevant_tokens["target"])}\n'
                        f'the result is: {str(relevant_tokens["output"])}')
        return relevant_token_ids

    def dataset_statistics(self,
                 processor,
                 data_set: str,
                 batch_size: int = 100,
                 disable_tqdm: bool = False):
        """
        Be able to reproduce the baselines from the published paper.
        In addition, it computes the number of Table 4: Query resolution datasets statistics.
        """
        self.model.eval()
        data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False, max_processes=1)
        data_loader = data_silo.get_data_loader(data_set)

        metrics = MicroMetrics()
        input_lengths = []
        number_of_positive_terms = []
        with torch.no_grad():
            progress_bar = tqdm(data_loader, disable=disable_tqdm)
            for step, batch in enumerate(progress_bar):
                for i in range(0, len(batch['input_ids'])):
                    relevant_tokens = self._get_classifier_result(input_token_ids=batch['input_ids'][i],
                                                                  attention_mask=batch['attention_mask'][i],
                                                                  target_tensor=batch['target'][i],
                                                                  output_tensor=batch['attention_mask'][i].type(torch.DoubleTensor),
                                                                  verbose=False)
                    metrics.add_to_values(predicted_tokens=relevant_tokens['input'],
                                          gold_tokens=relevant_tokens['target'],
                                          counts=False)

                    input_lengths.append(list(batch['start_of_words'][i]).count(1))
                    number_of_positive_terms.append(list(batch['target'][i]).count(1))

        print('\n')  # End the line printed by the progress_bar
        logger.info(f"Statistics of the '{data_set}' data set. \n"
                    f"Total queries:  {len(input_lengths)}\n"
                    f"Total terms:    {np.mean(input_lengths):.2f} +- {np.std(input_lengths):.2f}\n"
                    f"Positive terms: {np.mean(number_of_positive_terms):.2f} +- {np.std(number_of_positive_terms):.2f}")
        self._print_metrics(**metrics.calc_metric())

    def eval(self, data_loader, disable_tqdm: bool = False):
        self.model.to(self.device)
        self.model.eval()

        t0 = time.time()
        metrics = MicroMetrics()

        with torch.no_grad():
            progress_bar = tqdm(data_loader, disable=disable_tqdm)

            for step, batch in enumerate(progress_bar):
                # Calculate elapsed time in minutes.
                elapsed = datetime.timedelta(seconds=int(round(time.time() - t0)))
                progress_bar.set_description(f"Evaluating. Elapsed: {elapsed}")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     token_type_ids=batch['token_type_ids']
                                     )
                sigmoid_outputs = self.sigmoid(outputs).detach()
                for i in range(0, len(batch['input_ids'])):
                    relevant_tokens = self._get_classifier_result(input_token_ids=batch['input_ids'][i],
                                                                  attention_mask=batch['attention_mask'][i],
                                                                  target_tensor=batch['target'][i],
                                                                  output_tensor=sigmoid_outputs[i],
                                                                  verbose=False)
                    metrics.add_to_values(predicted_tokens=relevant_tokens['output'], gold_tokens=relevant_tokens['target'])
        print('\n')  # End the line printed by the progress_bar
        self._print_metrics(**metrics.calc_metric())

    def _print_metrics(self, micro_recall, micro_precision, micro_f1):
        logger.info(f"\nMicro recall   : {micro_recall:.5f} ({(micro_recall - self._prev_eval_results['r']):.5f})\n"
                    f"Micro precision: {micro_precision:.5f} ({(micro_precision - self._prev_eval_results['p']):.5f})\n"
                    f"Micro F1       : {micro_f1:.5f} ({(micro_f1 - self._prev_eval_results['f1']):.5f})\n")
        self._prev_eval_results['p'] = micro_precision
        self._prev_eval_results['r'] = micro_recall
        self._prev_eval_results['f1'] = micro_f1

    def rewrite(self, question: str, history):
        """
        Rewrite a single question + history ad hoc instead of using a data set.
        """
        # TODO
        pass