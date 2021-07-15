import datasets
from datasets import load_dataset
from haystack import Pipeline
from haystack.eval import EvalDocuments
from haystack.preprocessor.cleaning import remove_stopwords_and_stem
from haystack.query_rewriting.query_resolution_processor import QuretecProcessor
from haystack.query_rewriting.query_resolution import QueryResolution
from haystack.retriever.anserini import SparseAnseriniRetriever
import spacy
nlp = spacy.load("en_core_web_sm")

topics = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}
collection = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'test_collection_sample', split="test")

retriever = SparseAnseriniRetriever(huggingface_dataset=collection,
                                    huggingface_dataset_converter=lambda x: {'id': x.id, 'content': remove_stopwords_and_stem(x.text, nlp)},
                                    searcher_config={"Dirichlet": {'mu': 2500}})
processor = QuretecProcessor(tokenizer=None,
                             dataset_name=None,
                             train_split='0:4',  # Only the test split is required for evaluation
                             dev_split='0:4',  # Only the test split is required for evaluation
                             test_split='0:4',
                             max_seq_len=300)
qr = QueryResolution(pretrained_model_path="uva-irlab/quretec", processor=processor)
eval_retriever = EvalDocuments(top_k_eval_documents=1000, open_domain=False)

p = Pipeline()
p.add_node(component=qr, name="Rewriter", inputs=["Query"])
p.add_node(component=retriever, name="Retriever", inputs=["Rewriter"])
p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])

p.eval_qrels(eval_component_name='EvalRetriever',
             topics=[topic for qid, topic in enumerate(topics) if len(topic['history']) > 0],
             qrels=qrels,
             dump_results=True)
eval_retriever.print()
exit()