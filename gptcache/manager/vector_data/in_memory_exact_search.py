from typing import List

import numpy as np
from docarray.typing import NdArray
from pydantic import parse_obj_as

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_in_memory_exact_search

import_in_memory_exact_search()

from docarray import BaseDoc, DocList  # pylint: disable=C0413
from docarray.index import InMemoryExactNNIndex  # pylint: disable=C0413


class DocarrayVectorData(BaseDoc):
    id: int
    data: NdArray


class InMemoryExactSearch(VectorBase):
    """vector store: InMemoryExactSearch"""

    def __init__(self):
        self._index = InMemoryExactNNIndex[DocarrayVectorData]()

    def mul_add(self, datas: List[VectorData]):
        docs = DocList[DocarrayVectorData](
            DocarrayVectorData(id=data.id, data=data.data) for data in datas
        )
        self._index.index(docs)

    def search(self, data: np.ndarray, top_k: int = -1):
        query = parse_obj_as(NdArray, data)
        docs, scores = self._index.find(query, search_field="data", limit=top_k)
        return list(zip(scores, docs.id))

    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        del self._index[ids]
