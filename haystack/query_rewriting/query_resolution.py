import json
import re
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
from haystack.eval import EvalQueryResolution
from haystack.query_rewriting.data_handler import get_full_word
import spacy

logger = logging.getLogger(__name__)

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
                 dropout_prob=0.0,
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
        torch.save(self.state_dict(), f=Path(save_dir) / 'pytorch_model.bin')
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
                 model_args: dict = {},
                 space_lang="en_core_web_sm"
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

        if use_gpu and torch.cuda.is_available():
            logger.info("Using GPU Cuda")
            self.device = torch.device("cuda")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")

        self.nlp = spacy.load(space_lang)

    def train(self,
              processor,
              eval_metrics: EvalQueryResolution,
              eval_data_set: str,
              learning_rate: float = 3e-6,
              batch_size: int = 2,
              gradient_clipping: float = 1.0,
              n_gpu: int = 1,
              optimizer_name: str = 'AdamW',
              evaluate_every: int = 200,
              print_every: int = 200,
              epsilon: float = 1e-08,  # TODO QuReTec does not mention epsilon
              n_epochs: int = 1,
              save_dir: str = "saved_models",
              disable_tqdm: bool = False,
              grad_acc_steps: int = 2,
              weight_decay: float = 0.01,
              datasilo_args: dict = None,
              num_warmup_steps: int = 200,
              early_stopping: int = None,
              checkpoint_every: int = None,
              ):
        logger.info(f'Training QueryResolution with batch_size={batch_size}, gradient_clipping={gradient_clipping}, '
                    f'epsilon={epsilon}, n_gpu={n_gpu}, grad_acc_steps={grad_acc_steps}, evaluate_every={evaluate_every}, '
                    f'print_every={print_every}, early_stopping={early_stopping}')
        if datasilo_args is None:
            datasilo_args = {
                "caching": False,
            }

        if checkpoint_every is None and evaluate_every:
            checkpoint_every = evaluate_every

        self.data_silo = DataSilo(processor=processor,
                                  batch_size=batch_size,
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
            'dropout': self.model.dropout_prob
        }
        model_save_dir = Path(save_dir) / ("query_resolution_" + dict_to_string(params_dict))

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
                    output_tensor = self.sigmoid(outputs.detach()).detach()[0].float()
                    input_id_tensor = batch['input_ids'][0]
                    target_tensor = target[0]
                    self._print_classifier_result(input_token_ids=input_id_tensor,
                                                  attention_mask=batch['attention_mask'][0],
                                                  target_tensor=target_tensor,
                                                  output_tensor=output_tensor,
                                                  verbose=True)
                    predicted_tokens, gold_tokens = self._get_predicted_and_gold_tokens(
                        input_ids=input_id_tensor.tolist(),
                        predicted_tensor=output_tensor.round().tolist(),
                        target_tensor=target_tensor.tolist()
                    )
                    logger.info(f"The items used for metrics are: \n"
                                f"Predicted words: {predicted_tokens}\n"
                                f"Gold tokens: {gold_tokens}")
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
                    logger.info(f"Evaluating at step {step}")
                    self.eval(data_loader=self.data_silo.get_data_loader(eval_data_set), metrics=eval_metrics)

                    # Eval sets the model in eval mode, so set it to training mode again
                    self.model.train()

                    if eval_metrics.has_no_improvement():
                        break

                if early_stopping and step >= early_stopping:
                    break

                if checkpoint_every and step % checkpoint_every == 0:
                    checkpoint_model_save_dir = str(model_save_dir) + ".ckpt-" + str(step)
                    self.model.save(checkpoint_model_save_dir)
                    self.tokenizer.save_pretrained(save_directory=checkpoint_model_save_dir)

        self.model.save(model_save_dir)
        self.tokenizer.save_pretrained(save_directory=str(model_save_dir))

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

    def dataset_statistics(self,
                 processor,
                 metrics: EvalQueryResolution,
                 data_sets: List[str],
                 batch_size: int = 100,
                 disable_tqdm: bool = False,
                 debug: bool = True):
        """
        Be able to reproduce the baselines from the published paper.
        In addition, it computes the Original (all) of Table 4: Query resolution datasets statistics.
        """
        self.model.eval()
        data_silo = DataSilo(processor=processor,
                             batch_size=batch_size,
                             distributed=False,
                             max_processes=1,
                             caching=True,
                             automatic_loading=False)

        for data_set in data_sets:
            data_loader = data_silo.get_data_loader(data_set)

            input_lengths = []
            number_of_positive_terms = []
            with torch.no_grad():
                progress_bar = tqdm(data_loader, disable=disable_tqdm)
                progress_bar.set_description(f"Computing dataset statistics of the '{data_set}' data set")

                for step, batch in enumerate(progress_bar):
                    for i in range(0, len(batch['input_ids'])):
                        target_tensor = batch['target'][i]
                        input_ids_tensor = batch['input_ids'][i]
                        attention_tensor = batch['attention_mask'][i]
                        # Provide start of words to simulate Original (all) from Table 5 in original paper
                        predicted_tensor = batch['start_of_words'][i]

                        predicted_tokens, gold_tokens = self._get_predicted_and_gold_tokens(input_ids=input_ids_tensor.tolist(),
                                                                                            predicted_tensor=predicted_tensor.tolist(),
                                                                                            target_tensor=target_tensor.tolist())
                        metrics.run(predicted_tokens=predicted_tokens, gold_tokens=gold_tokens)

                        input_lengths.append(list(batch['start_of_words'][i]).count(1))
                        number_of_positive_terms.append(list(batch['target'][i]).count(1))

                        if debug and step == 0 and i == 0:
                            self._print_classifier_result(input_token_ids=input_ids_tensor,
                                                          target_tensor=target_tensor,
                                                          output_tensor=predicted_tensor,
                                                          attention_mask=attention_tensor,
                                                          verbose=True)
                            logger.info(f"The items used for metrics are: \n"
                                        f"Predicted words: {predicted_tokens}\n"
                                        f"Gold tokens: {gold_tokens}")

            print('\n')  # End the line printed by the progress_bar
            logger.info(f"Statistics of the '{data_set}' data set. \n"
                        f"Total queries:  {len(input_lengths)}\n"
                        f"Total terms:    {np.mean(input_lengths):.2f} +- {np.std(input_lengths):.2f}\n"
                        f"Positive terms: {np.mean(number_of_positive_terms):.2f} +- {np.std(number_of_positive_terms):.2f}")
            metrics.print(mode="query_resolution", use_logged_values=False)
            metrics.init_counts()

    def _get_predicted_and_gold_tokens(self, input_ids, predicted_tensor, target_tensor):
        predicted_terms = []
        gold_terms = []
        for idx, token_id in enumerate(input_ids):
            if predicted_tensor[idx] == 1.0 or target_tensor[idx] == 1.0:
                # Paper says: "We apply lowercase, lemmatization and stopword removal to qi∗,
                # q1:i−1 and qi using Spacy12 before calculating term overlap in Equation 2.", so
                # lemmatise the full word and remove non-stopwords
                full_word, _, _, parsed = get_full_word(tokenizer=self.tokenizer,
                                                        input_ids=input_ids,
                                                        pos=idx,
                                                        parse=True)

                # Check if the non-lemmatised full word is in stop words, since the Spacy stopwords set
                # does not contain lemmatised words.
                if parsed.is_stop is False and parsed.is_punct is False:
                    if predicted_tensor[idx] == 1.0:
                        predicted_terms.append(parsed.lemma_)
                    if target_tensor[idx] == 1.0:
                        gold_terms.append(parsed.lemma_)
        return predicted_terms, gold_terms

    def eval(self,
             data_loader,
             metrics: EvalQueryResolution,
             disable_tqdm: bool = False):
        self.model.to(self.device)
        self.model.eval()

        t0 = time.time()

        with torch.no_grad():
            progress_bar = tqdm(data_loader, disable=disable_tqdm)

            for step, batch in enumerate(progress_bar):
                # Calculate elapsed time in minutes.
                elapsed = datetime.timedelta(seconds=int(round(time.time() - t0)))
                progress_bar.set_description(f"Evaluating in progress. Elapsed: {elapsed}")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     token_type_ids=batch['token_type_ids']
                                     )
                sigmoid_outputs = self.sigmoid(outputs).detach()
                for i in range(0, len(batch['input_ids'])):
                    predicted_tokens, gold_tokens = self._get_predicted_and_gold_tokens(input_ids=batch['input_ids'][i].tolist(),
                                                                                        predicted_tensor=sigmoid_outputs[i].float().round().tolist(),
                                                                                        target_tensor=batch['target'][i].tolist()
                                                                                        )
                    metrics.run(predicted_tokens=predicted_tokens, gold_tokens=gold_tokens)
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