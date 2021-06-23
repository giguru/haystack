import argparse
import sys
import os
import logging
import torch

from transformers import BertConfig

from haystack import Pipeline

sys.path.append('..')
from haystack.query_rewriting.data_handler import CanardProcessor
from haystack.query_rewriting.query_resolution import QueryResolution


logger = logging.getLogger(__name__)
logger.info(f'{torch.cuda.is_available()} {torch.cuda.device_count()}')


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--pretrained_model_path",
                    default=None,
                    help="Use existing saved model")
args = parser.parse_args()


def train():
    config = BertConfig.from_pretrained("bert-large-uncased",
                                        num_labels=len(CanardProcessor.get_labels()),
                                        finetuning_task="ner",
                                        hidden_dropout_prob=0.4,
                                        label2id=CanardProcessor.label2id(),
                                        id2label=CanardProcessor.id2label(),
                                        pad_token_id=CanardProcessor.pad_token_id()
                                        )
    query_resolution = QueryResolution(config=config,
                                       use_gpu=True,
                                       model_args={
                                            'bert_model': "bert-large-uncased",
                                            'max_seq_len': 300,
                                       })
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                # train_split=4, test_split=4, dev_split=4,
                                max_seq_len=300)
    query_resolution.train(processor,
                           evaluate_every=2500,
                           datasilo_args={"caching": False},
                           learning_rate=args.learning_rate)
    query_resolution.eval(processor)


def evaluate():
    query_resolution = QueryResolution(pretrained_model_path=args.pretrained_model_path)
    processor = CanardProcessor(tokenizer=query_resolution.tokenizer,
                                train_split=4, test_split=None, dev_split=4,
                                max_seq_len=300)
    query_resolution.eval(processor)

evaluate()
exit()