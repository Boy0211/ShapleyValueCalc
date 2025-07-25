from functools import wraps
import time

def timer_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        end = time.perf_counter()

        if not hasattr(self, 'timings'):
            self.timings = {}

        if func.__name__ not in self.timings:
            self.timings[func.__name__] = end - start
        else:
            self.timings[func.__name__] += end - start

        return result
    
    return wrapper