from typing import List


def recallk(actual: List[int], pred: List[int], k: int = 25) -> float:
    """recall k

    Parameters
    ----------
    actual : List[int]
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
    set_actual = set(actual)
    recall_k = len(set_actual & set(pred[:k])) / min(k, len(set_actual))
    return recall_k


def unique(sequence: List[int]) -> List[int]:
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


# ndcgk 관련해서 README에 수식 정리
