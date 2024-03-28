"""
Benchmark the import of fastdfe
"""

import time

start = time.time()
# noinspection PyUnresolvedReferences
import fastdfe

end = time.time()

print(f"fastdfe imported in {end - start:.2f} seconds")
