from typing import List

import numpy as np


def recallk(true: List[int], pred: List[int], k: int = 25) -> float:
    """recall k

    Parameters
    ----------
    true : List[int]
        true item
    pred : List[int]
        prediction item
    k : int, optional
        top k, by default 25

    Returns
    -------
    float
        recall k
    """
    set_true = set(true)
    recall_k = len(set_true & set(pred[:k])) / min(k, len(set_true))
    return recall_k


def get_order_preserved_unique(sequence: List[int]) -> List[int]:
    """unique (preserve list item order)

    Parameters
    ----------
    sequence : List[int]
        item list to be unique

    Returns
    -------
    List[int]
        (preserved order) unique item list
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def ndcgk(true: List[int], pred: List[int], k: int = 25) -> float:
    """NDCG@K

    Parameters
    ----------
    true : List[int]
        _description_
    pred : List[int]
        _description_
    k : int, optional
        _description_, by default 25

    Returns
    -------
    float
        ndcg_k value
    """
    set_true = set(true)
    idcg = sum([1.0 / np.log(i + 2) for i in range(min(k, len(set_true)))])
    dcg = 0.0
    unique_pred = get_order_preserved_unique(pred[:k])
    for i, r in enumerate(unique_pred):
        if r in set_true:
            dcg += 1.0 / np.log(i + 2)
    ndcg_k = dcg / idcg
    return ndcg_k
