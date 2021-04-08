from pathlib import Path
from typing import Any

import torch
from farm.modeling.adaptive_model import BaseAdaptiveModel
from tqdm import tqdm
import time
from torch import nn
import datetime
import shutil

from farm.modeling.tokenization import Tokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.data_handler.data_silo import DataSilo

from haystack import BaseComponent
from haystack.query_rewriting.data_handler import QuacProcessor


class QueryResolutionModel(nn.Module):
    config_file_name = "query_resolution_model_config.json"
    model_binary_file_name = "query_resolution.bin"


    def __init__(self,
                 model_name_or_path: str,
                 linear_layer_size=1024,  # 1024 is suitable for the bert-large-uncased model from HuggingFace Transformers library
                 dropout_prob=0.0  # probability of an element to be zeroed
                 ):
        """

        """
        super(QueryResolutionModel, self).__init__()
        self._linear_layer_size = linear_layer_size
        self._bert = AutoModel.from_pretrained(model_name_or_path)
        self._dropout = nn.Dropout(p=dropout_prob)

        # 512
        self._linear1 = nn.Linear(self._linear_layer_size, 512)
        # Since the output of this model should be a classification token for each term
        # 0 for unrelevant tokens and 1 for relevant tokens.The sigmoid function makes sure
        # this value is between zero and 1.
        self._sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        bert_output = self._bert(**kwargs)
        sequence_output = bert_output.last_hidden_state

        # sequence_output has the following shape: (batch_size, sequence_length, 512)
        # extract the 1st token's embeddings
        dropout_output = self._dropout(sequence_output[:, 0, :].view(-1, self._linear_layer_size))
        linear1_output = self._linear1(dropout_output)
        results = self._sigmoid(linear1_output)

        return results

    def save_config(self, save_dir):
        save_filename = Path(save_dir) / QueryResolutionModel.config_file_name
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        # Save Weights
        save_name = Path(save_dir) / QueryResolutionModel.model_binary_file_name
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        torch.save(model_to_save.state_dict(), save_name)
        self.save_config(save_dir)


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

        # Set derived instance properties
        self.tokenizer = Tokenizer.load(model_name_or_path, **tokenizer_args)
        self.model = QueryResolutionModel(model_name_or_path=model_name_or_path, **model_args)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def train(self,
              processor,
              learning_rate: float = 2e-5,
              batch_size: int = 4,
              gradient_clipping: float = 1.0,
              n_gpu: int = 1,
              optimizer_name: str = 'TransformersAdamW',
              evaluate_every: int = 1000,
              num_warmup_steps: int = 100,
              epsilon: float = 1e-08,  # TODO QuReTec does not mention epsilon
              n_epochs: int = 3,  # TODO QuReTec paper does not mention epochs.
              save_dir: str = "../saved_models/query_resolution",
              disable_tqdm: bool=False,
              checkpoint_every: int = 1000,
              ):

        data_silo = DataSilo(processor=processor,
                             batch_size=batch_size,
                             distributed=False,
                             max_processes=1)

        # Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={"name": optimizer_name,
                            "correct_bias": False,
                            "weight_decay": 0.01,
                            "eps": epsilon
                            },
            schedule_opts={"name": "LinearWarmup",
                           "num_warmup_steps": num_warmup_steps
                           },
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            device=self.device
        )

        # Set in training mode
        self.model.train()

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        loss = 0

        for epoch in range(n_epochs):
            t0 = time.time()
            train_data_loader = data_silo.get_data_loader("train")

            progress_bar = tqdm(train_data_loader, disable=disable_tqdm)
            for step, batch in enumerate(progress_bar):
                # Calculate elapsed time in minutes.
                elapsed = datetime.timedelta(seconds=int(round(time.time() - t0)))
                progress_bar.set_description(f"Train epoch {epoch + 1}/{n_epochs} (Cur. train loss: {loss:.4f}) Elapsed {elapsed}")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Always clear previous gradients before performing backward pass. PyTorch doesn't do this automatically
                # accumulating the gradients is "convenient while training RNNs"
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     token_type_ids=batch['token_type_ids']
                                     )
                target = batch['target'].float()
                loss = loss_fn(outputs, target)
                print(' ')
                loss.backward()

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)

                # Update model parameters
                optimizer.step()
                optimizer.zero_grad()  # TODO is this necessary when using model.zero_grad()?

                # Update the learning rate
                lr_schedule.step()

                # TODO save every x points
                # if checkpoint_every and step % checkpoint_every == 0:
                    
        self.model.save(Path(save_dir))

    def run(self, *args: Any, **kwargs: Any):
        pass

    def rewrite(self, question: str, history):
        pass