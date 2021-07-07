import argparse
import logging
import sys
import pyterrier as pt
import datasets
from haystack import Pipeline
from haystack.document_store import InMemoryDocumentStore
from haystack.query_rewriting.data_handler import CanardProcessor
from haystack.ranker import FARMRanker
from haystack.reader import FARMReader
from haystack.retriever import TerrierRetriever

sys.path.append('..')
from haystack.query_rewriting.query_resolution import QueryResolution

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



if not pt.started():
    logger.info("Started ")
    pt.init()

# Load data sets
topics = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}
collection = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', split="test")

# Load pipeline elements
retriever = TerrierRetriever(huggingface_dataset=collection,
                             top_k=1000,
                             config_json={'wmodel': 'BM25F'})
processor = CanardProcessor(tokenizer=None,
                            dataset_name=None,
                            train_split='0:4',  # Only the test split is required for evaluation
                            dev_split='0:4',  # Only the test split is required for evaluation
                            test_split='0:4',
                            max_seq_len=300)
qr = QueryResolution(pretrained_model_path="uva-irlab/quretec", processor=processor)
ranker = FARMRanker(model_name_or_path="bert-large-uncased")
# reader = FARMReader(model_name_or_path="ber")


# Build pipeline
conversational_pipeline = Pipeline()
conversational_pipeline.add_node(component=qr, name="Rewriter", inputs=["Query"])
conversational_pipeline.add_node(component=retriever, name="Retriever", inputs=["Rewriter"])
# conversational_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])

results = {}
for qid, topic in enumerate(topics):
    if len(topic['history']) > 0:
        id = topic['qid']
        result = conversational_pipeline.run(query=topic['query'], history=" ".join(topic['history']), id=id)
        results[id] = result['documents']

print(results)