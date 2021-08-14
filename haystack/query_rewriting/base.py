from abc import ABC
from functools import wraps
from time import perf_counter


class BaseReformulator(ABC):
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