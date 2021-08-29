from haystack import Pipeline
from haystack.eval import EvalDocuments
from haystack.query_rewriting.transformer import GenerativeReformulator, ClassificationReformulator
from haystack.retriever.anserini import SparseAnseriniRetriever
import datasets

## Load datasets
topics = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")

# Convert into the right data format
topics = [topic for qid, topic in enumerate(topics)]
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}

# Load components

# You can either use the ClassifcationReformulator with the 'uva-irlab/quretec' model...
# reformulator = ClassificationReformulator(pretrained_model_path="uva-irlab/quretec")
# or use the GenerativeReformulator with any Seq2SeqLM model
reformulator = GenerativeReformulator(pretrained_model_path="castorini/t5-base-canard")

retriever = SparseAnseriniRetriever(prebuilt_index_name='cast2019', searcher_config={"BM25": {}})
eval_retriever = EvalDocuments(top_k_eval_documents=1000, open_domain=False)

# Build pipeline
p = Pipeline()
p.add_node(component=reformulator, name="Reformulator", inputs=["Query"])
p.add_node(component=retriever, name="Retriever", inputs=["Reformulator"])
p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])

# Do evaluation
p.eval_qrels(eval_component_name="EvalRetriever",
             qrels=qrels,
             topics=topics,
             dump_results=True)

# Print metric results
eval_retriever.print()

# Many components register execution time, so you can print the total execution times
reformulator.print_time()
retriever.print_time()