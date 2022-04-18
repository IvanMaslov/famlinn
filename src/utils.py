import time


class Perf:

    def __init__(self, message):
        self.start = None
        self.finish = None
        self.message = message

    def __enter__(self):
        self.start = time.time_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish = time.time_ns()

    def __del__(self):
        print(self.message, f"(millis={(self.finish - self.start) / 1_000_000})")
