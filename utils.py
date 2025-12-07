import numpy as np

# Returns euclidean distance between two embeddings
def get_distance(embedding1, embedding2):
    """
    get_distance function
    Function that returns euclidean distance between
    two embeddings.

    Args:
        embedding1: first embedding.
        embedding2: second embedding.

    Returns:
        The euclidean distance value.

    Raises:
        ValueError: if embedding1 or embedding2 is not a 1D array or if they have different shapes.
    """
    if embedding1.ndim != 1:
        raise ValueError("embedding1 must be a 1D array. Got shape {}".format(embedding1.shape))
    if embedding2.ndim != 1:
        raise ValueError("embedding2 must be a 1D array. Got shape {}".format(embedding2.shape))
    if(embedding1.shape != embedding2.shape):
        raise ValueError("embedding1 and embedding2 must have the same shape. Got shapes {} and {}".format(embedding1.shape, embedding2.shape))

    total = np.sum(np.square(embedding2 - embedding1))
    return np.sqrt(total)