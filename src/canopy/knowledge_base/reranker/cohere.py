from copy import deepcopy
from typing import List, Optional

try:
    import cohere
except (OSError, ImportError, ModuleNotFoundError) as e:
    _cohere_installed = False
else:
    _cohere_installed = True

from canopy.knowledge_base.models import KBQueryResult
from canopy.knowledge_base.reranker import Reranker


class CohereReranker(Reranker):
    """
    Reranker that uses Cohere's text embedding to rerank documents.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = 'rerank-english-v2.0',
                 top_n: int = 10,
                 threshold: float = 0.0):
        """
        Initialize the Cohere reranker.

        Args:
            api_key: The Cohere API key. If not provided, the API key will be read from the COHERE_API_KEY environment variable.
        """
        if not _cohere_installed:
            raise ImportError(
                "Failed to import cohere. Make sure you install cohere extra "
                "dependencies by running: "
                "`pip install canopy-sdk[cohere]"
            )

        self._client = cohere.Client(api_key=api_key)
        self._model_name = model_name
        self._top_n = int(top_n)
        self._threshold = float(threshold)

    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        reranked: List[KBQueryResult] = []
        for q_res in results:
            texts = [doc.text for doc in q_res.documents]
            response = self._client.rerank(query=q_res.query,
                                           documents=texts,
                                           top_n=self._top_n,
                                           model=self._model_name, )
            reranked_docs = []
            for r in response:
                if r.relevance_score >= self._threshold:
                    reranked_docs.append(deepcopy(q_res.documents[r.index]))
                    reranked_docs[-1].score = r.relevance_score

            reranked.append(KBQueryResult(query=q_res.query,
                                          documents=reranked_docs))
        return reranked

    async def arerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        raise NotImplementedError()
