from typing import List, Dict
from datasets import Dataset
from pyserini.index._base import JGenerators
from pyserini.search import SimpleSearcher
from pyserini.index import Generator
from haystack import Document
from haystack.retriever.base import BaseRetriever
import logging
import os
import json

logger = logging.getLogger(__name__)


class SparseAnseriniRetriever(BaseRetriever):
    def __init__(self,
                 searcher_config: Dict[str, dict],
                 top_k: int = 1000,
                 prebuilt_index_name: str = None,
                 huggingface_dataset: Dataset = None,
                 huggingface_dataset_converter = None
                 ):
        """
        @param prebuilt_index_name: str
            E.g. robust04, msmarco-passage-slim, msmarco-passage or cast2019. For all available prebuilt indexes,
            please call pyserini.SimpleSearcher.list_prebuilt_indexes() or search on Google.
        """
        if prebuilt_index_name is not None:
            self.searcher = SimpleSearcher.from_prebuilt_index(prebuilt_index_name)
        elif huggingface_dataset is not None:
            self.index_path = self._build_index_using_huggingface(huggingface_dataset, huggingface_dataset_converter)
            self.searcher = SimpleSearcher(self.index_path)
        else:
            raise ValueError('Please provide either a prebuilt_index_name or huggingface_dataset.')
        for key, params in searcher_config.items():
            if key == 'Dirichlet':
                self.searcher.set_qld(**params)
            elif key == 'BM25':
                self.searcher.set_bm25(**params)
            elif key == 'RM3':
                self.searcher.set_rm3(**params)
            elif key in self.searcher:
                getattr(self.searcher, key)(**params)
            else:
                raise KeyError("Invalid key in `searcher_config`. The allowed keys are: Dirichlet, BM25, RM3 or a function of SimpleSearcher")

        self.top_k = top_k

    def _build_index_using_huggingface(self, dataset: Dataset, converter = None):
        index_path = './huggingface_' + dataset.info.builder_name
        if os.path.isfile(index_path):
            logger.info(f"Using existing Lucene index: {index_path}")
        else:
            logger.info('Preparing .jsonl for indexing Anserini')
            temp_jsonl_file = './temp.jsonl'
            if os.path.isfile(temp_jsonl_file):
                os.remove(temp_jsonl_file)

            with open(f'{temp_jsonl_file}') as writer:
                n_empty_documents = []
                for doc in dataset:
                    doc_to_index = converter(doc) if converter else doc
                    if 'id' not in doc_to_index:
                        raise KeyError(f'Each document should have an ID! Object: {doc_to_index}')
                    if 'content' not in doc_to_index or not doc_to_index['content']:
                        n_empty_documents.append(doc_to_index['id'])
                    else:
                        writer.write(json.dump(doc_to_index, separators=(',', ':')) + '\n')

                if len(n_empty_documents) > 0:
                    logger.info(f"{len(n_empty_documents)} documents were not indexed: {', '.join(n_empty_documents)}")

                from jnius import autoclass
                JIndexCollection = autoclass('io.anserini.index.IndexCollection')
                JIndexCollection.main({
                    'generator': JGenerators.DefaultLuceneDocumentGenerator.value,
                    'collection': 'JsonCollection',
                    'input': temp_jsonl_file,
                    'index': index_path
                })

            os.remove(temp_jsonl_file)
        return index_path

    def retrieve(self, **kwargs) -> List:
        query = kwargs.get('query', None)  # type: str
        top_k = kwargs.get('top_k', None) or self.top_k  # type: int

        hits = self.searcher.search(q=query, k=top_k)
        results = []
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            results.append(Document(id=hit.docid,
                                    score=hit.score,
                                    text=doc.contents()))
        return results