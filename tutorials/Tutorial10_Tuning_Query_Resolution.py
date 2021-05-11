import sys

from farm.data_handler.data_silo import DataSilo
from farm.modeling.tokenization import Tokenizer

sys.path.append('..')
from haystack.query_rewriting.data_handler import CanardProcessor
from haystack.query_rewriting.query_resolution import QueryResolution


def find_hyperparameters():
    dropouts = [
        0.1, 0.2,
        0.3, 0.4
    ]
    lrs = [2e-5, 3e-5, 3e-6]
    for dropout_prod in dropouts:
        for lr in lrs:
            query_resolution = QueryResolution(model_args={'dropout_prob': dropout_prod}, use_gpu=True)
            processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                        max_seq_len=512,
                                        train_split=6000,
                                        dev_split=None,
                                        test_split=10)
            query_resolution.train(processor,
                                   n_gpu=1,
                                   print_every=200,
                                   evaluate_every=200,
                                   eval_data_set="dev",
                                   learning_rate=lr)


def main():
    query_resolution = QueryResolution(model_args={'dropout_prob': 0.1}, use_gpu=True)
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=None,
                                test_split=None,
                                dev_split=None)
    query_resolution.train(processor,
                           print_every=200,
                           evaluate_every=200,
                           eval_data_set="dev")
    query_resolution.eval(query_resolution.data_silo.get_data_loader('test'))


def data_set_statistics():
    query_resolution = QueryResolution(use_gpu=True)
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=10,
                                dev_split=None,
                                test_split=10)
    query_resolution.dataset_statistics(processor, data_set="dev")


def evaluate():
    query_resolution = QueryResolution(model_name_or_path="./saved_models/query_resolution_learning_rate_3e-05_eps_1e-08_weight_decay_0_01_dropout_0_1",
                                       # tokenizer_args={
                                       #     'tokenizer_class': 'BertTokenizer'
                                       # }
                                       )
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                max_seq_len=512,
                                train_split=10,
                                dev_split=10,
                                test_split=None)
    data_silo = DataSilo(processor=processor, batch_size=100, distributed=False, max_processes=1)
    query_resolution.eval(data_silo.get_data_loader('dev'))


evaluate()
exit()