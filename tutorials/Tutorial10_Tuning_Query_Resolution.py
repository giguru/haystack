from haystack.query_rewriting.data_handler import QuacProcessor
from haystack.query_rewriting.query_resolution import QueryResolution


query_resolution = QueryResolution(model_args={'dropout_prob': 0.1})

processor = QuacProcessor(tokenizer=query_resolution.tokenizer,
                          max_seq_len=512  # 512, because that is the maximum number of tokens BERT uses
                          )
query_resolution.train(processor)

