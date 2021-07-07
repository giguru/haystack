import argparse
import logging
import sys

from haystack.ranker import FARMRanker

sys.path.append('..')

logger = logging.getLogger(__name__)

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

ranker = FARMRanker(model_name_or_path="bert-large-uncased")