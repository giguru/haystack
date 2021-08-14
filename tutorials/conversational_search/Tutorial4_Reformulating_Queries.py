from haystack import Pipeline
from haystack.eval import EvalDocuments
from haystack.query_rewriting.query_resolution import QueryResolution
from haystack.query_rewriting.query_resolution_processor import QuretecProcessor
from haystack.query_rewriting.transformer import TransformerReformulator
from haystack.retriever.anserini import SparseAnseriniRetriever
import datasets

# Load datasets
topics = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}

# Several methods of query rewriting are possible
processor = QuretecProcessor(tokenizer=None,
                             dataset_name=None,
                             train_split='0:4',  # Only the test split is required for evaluation
                             dev_split='0:4',  # Only the test split is required for evaluation
                             test_split='0:4',
                             max_seq_len=300)
qr = QueryResolution(pretrained_model_path="uva-irlab/quretec", processor=processor)
ntr = TransformerReformulator(pretrained_model_path="castorini/t5-base-canard")

# Load other components
retriever = SparseAnseriniRetriever(prebuilt_index_name='cast2019', searcher_config={"Dirichlet": {'mu': 2500}})
eval_retriever = EvalDocuments(top_k_eval_documents=1000, open_domain=False)

for reformulator in [qr, ntr]:
    eval_retriever.init_counts()
    p = Pipeline()
    p.add_node(component=qr, name="Rewriter", inputs=["Query"])
    p.add_node(component=retriever, name="Retriever", inputs=["Rewriter"])
    p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Rewriter"])
    p.eval_qrels(eval_component_name="EvalRetriever",
                 qrels=qrels,
                 topics=[topic for qid, topic in enumerate(topics) if len(topic['history']) > 0],
                 dump_results=True)