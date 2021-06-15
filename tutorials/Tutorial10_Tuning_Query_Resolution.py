import sys

from farm.data_handler.data_silo import DataSilo
import logging
import torch

from transformers import BertConfig

from haystack.eval import EvalQueryResolution

sys.path.append('..')
from haystack.query_rewriting.data_handler import CanardProcessor
from haystack.query_rewriting.query_resolution import QueryResolution


logger = logging.getLogger(__name__)
logger.info(f'{torch.cuda.is_available()} {torch.cuda.device_count()}')


def find_hyperparameters():
    dropouts = [
        0.3, 0.4,
        0.1, 0.2,
    ]
    lrs = [3e-6, 1e-6, 3e-7]
    logger.info(f"Doing a hyperparameter search for dropouts={dropouts}, learning_rates={lrs}")
    for dropout_prod in dropouts:
        for lr in lrs:
            try:
                query_resolution = QueryResolution(model_args={'dropout_prob': dropout_prod}, use_gpu=True)
                processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                            max_seq_len=512,
                                            train_split=None,
                                            dev_split=None,
                                            test_split=None)
                query_resolution.train(processor,
                                       eval_metrics=EvalQueryResolution(use_counts=True),
                                       n_gpu=1,
                                       print_every=100,
                                       evaluate_every=100,
                                       eval_data_set="dev",
                                       learning_rate=lr,
                                       datasilo_args={
                                           "caching": False
                                       })
            except ZeroDivisionError as e:
                logger.info(e)


def train():
    config = BertConfig.from_pretrained("bert-large-uncased",
                                        num_labels=len(CanardProcessor.get_labels()) + 1,
                                        finetuning_task="ner",
                                        hidden_dropout_prob=0.4,)
    query_resolution = QueryResolution(config=config,
                                       use_gpu=True,
                                       model_args={
                                            'bert_model': "bert-large-uncased",
                                            'max_seq_len': 300,
                                       })
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer, max_seq_len=300, train_split=5, test_split=5, dev_split=5)
    query_resolution.train(processor,
                           eval_metrics=EvalQueryResolution(use_counts=True),
                           print_every=100,
                           evaluate_every=100,
                           eval_data_set="dev",
                           datasilo_args={"caching": False},
                           learning_rate=1e-6,
                           num_warmup_steps=100,
                           early_stopping=1200)

    logger.info("Evaluating test dataset...")
    query_resolution.eval(query_resolution.data_silo.get_data_loader('test'),
                          metrics=EvalQueryResolution(use_counts=True),
                          label_list=processor.get_labels())


def data_set_statistics():
    query_resolution = QueryResolution(use_gpu=True)
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=None,
                                dev_split=None,
                                test_split=None,
                                )
    query_resolution.dataset_statistics(processor,
                                        data_sets=["test"],
                                        metrics=EvalQueryResolution(use_counts=True)
                                        )


def evaluate():
    query_resolution = QueryResolution()
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=0,
                                dev_split=10,
                                test_split=None)
    data_silo = DataSilo(processor=processor, batch_size=100, distributed=False, max_processes=1)
    query_resolution.eval(data_silo.get_data_loader('test'), metrics=EvalQueryResolution(use_counts=True))


train()
exit()