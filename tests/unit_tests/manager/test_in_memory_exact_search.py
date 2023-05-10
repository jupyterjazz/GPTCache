import unittest

import numpy as np

from gptcache.manager.vector_data.base import VectorData
from gptcache.manager.vector_data.in_memory_exact_search import \
    InMemoryExactSearch


class TestInMemoryExactSearch(unittest.TestCase):
    def test_normal(self):
        index = InMemoryExactSearch()
        size = 1000
        dim = 512
        top_k = 10

        data = np.random.randn(size, dim).astype(np.float32)
        index.mul_add(
            [VectorData(id=i, data=v) for v, i in zip(data, list(range(size)))]
        )
        self.assertEqual(len(index.search(data[0], top_k)), top_k)
        index.mul_add([VectorData(id=size, data=data[0])])
        ret = index.search(data[0], top_k)
        self.assertIn(ret[0][1], [0, size])
        self.assertIn(ret[1][1], [0, size])
