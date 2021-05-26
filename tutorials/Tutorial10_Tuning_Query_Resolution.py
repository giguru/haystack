import sys

from farm.data_handler.data_silo import DataSilo
import logging
from haystack.eval import EvalQueryResolution

sys.path.append('..')
from haystack.query_rewriting.data_handler import CanardProcessor
from haystack.query_rewriting.query_resolution import QueryResolution


logger = logging.getLogger(__name__)


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
                                            test_split=None,
                                            include_current_turn_in_attention=False)
                query_resolution.train(processor,
                                       eval_metrics=EvalQueryResolution(use_counts=True),
                                       n_gpu=1,
                                       print_every=100,
                                       evaluate_every=100,
                                       eval_data_set="dev",
                                       learning_rate=lr,
                                       datasilo_args={
                                           "caching": True
                                       })
            except ZeroDivisionError as e:
                logger.info(e)


def train():
    query_resolution = QueryResolution(model_args={'dropout_prob': 0.2}, use_gpu=True)
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=None,
                                test_split=None,
                                dev_split=None,
                                include_current_turn_in_attention=False)
    query_resolution.train(processor,
                           eval_metrics=EvalQueryResolution(use_counts=True),
                           print_every=100,
                           evaluate_every=100,
                           eval_data_set="dev",
                           datasilo_args={
                               "caching": True
                           },
                           learning_rate=1e-6,
                           num_warmup_steps=0,
                           early_stopping=1200)

    logger.info("Evaluating test dataset...")
    query_resolution.eval(query_resolution.data_silo.get_data_loader('test'),
                          metrics=EvalQueryResolution(use_counts=True))


def data_set_statistics():
    query_resolution = QueryResolution(use_gpu=True)
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=None,
                                dev_split=None,
                                test_split=None,
                                include_current_turn_in_attention=True,
                                )
    query_resolution.dataset_statistics(processor,
                                        data_sets=["test"],
                                        metrics=EvalQueryResolution(use_counts=True)
                                        )


def evaluate():
    query_resolution = QueryResolution(model_name_or_path="bert-large-uncased")
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=0,
                                dev_split=10,
                                test_split=None)
    data_silo = DataSilo(processor=processor, batch_size=100, distributed=False, max_processes=1)
    query_resolution.eval(data_silo.get_data_loader('test'), metrics=EvalQueryResolution(use_counts=True))

train()
exit()