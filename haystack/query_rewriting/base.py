from abc import ABC, abstractmethod
from functools import wraps
from time import perf_counter
from typing import Any, Union, List

import torch

from haystack import BaseComponent


class BaseReformulator(BaseComponent):

    def __init__(self, use_gpu: bool = True):
        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
            self.n_gpu = torch.cuda.device_count()
        else:
            device = 'cpu'
            self.n_gpu = 1
        self.device = torch.device(device)

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions. """

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if attr_name not in self.__dict__:
                self.__dict__[attr_name] = 0
            tic = perf_counter()
            ret = fn(*args, **kwargs)
            toc = perf_counter()
            self.__dict__[attr_name] += toc - tic
            return ret

        return wrapper

    @abstractmethod
    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        pass

    def run(self, **kwargs: Any):
        run_query_timed = self.timing(self.run_query, "query_time")
        output, stream = run_query_timed(**kwargs)
        return {**kwargs, **output}, stream