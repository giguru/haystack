import logging
import torch
from typing import Any, List, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from haystack.query_rewriting.base import BaseReformulator

logger = logging.getLogger(__name__)


__all__ = ['TransformerReformulator']


class TransformerReformulator(BaseReformulator):
    outgoing_edges = 1

    def __init__(self,
                 pretrained_model_path: str,
                 max_length: int = 64,
                 num_beams: int = 10,
                 use_gpu: bool = True,
                 tokenizer_args: dict = {},
                 model_args: dict = {},
                 early_stopping: bool = True,
                 history_separator: str = '|||'
                 ):
        """
        Reformulate queries using transformers.
        Since AutoModel and AutoTokenizer

        Combinations that can be used are:
        - pretrained_model_path='castorini/t5-base-canard',
          transformer_class=T5ForConditionalGeneration,
          tokenizer_class=T5Tokenizer


        @param pretrained_model_path:
        @param max_length:
        @param num_beams:
        @param use_gpu:
        @param tokenizer_args:
        @param model_args:
        """

        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
            self.n_gpu = torch.cuda.device_count()
        else:
            device = 'cpu'
            self.n_gpu = 1

        self.max_length = max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.history_separator = history_separator

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, **tokenizer_args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path, **model_args)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        original_query = query
        if len(history) > 0:
            src_text = f" {self.history_separator} ".join(history) if isinstance(history, list) else history
            src_text = f"{src_text} {self.history_separator} {query}"
            input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=self.early_stopping,
            )
            rewritten_query = self.tokenizer.decode(
                output_ids[0, 0:],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            output = {
                "query": rewritten_query,
                "original_query": original_query,
                **kwargs
            }
        else:
            output = {
                "query": query,
                "original_query": query,
            }

        return output, "output_1"

    def run(self, pipeline_type, **kwargs: Any):
        if pipeline_type == "Query":
            run_query_timed = self.timing(self.run_query, "query_time")
            output, stream = run_query_timed(**kwargs)
        else:
            raise Exception(f"Invalid pipeline_type '{pipeline_type}'.")
        return output, stream




