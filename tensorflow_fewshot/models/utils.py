from tensorflow import expand_dims, sqrt, reduce_sum, square

def euclidean_distance(prototypes, embeddings):
    """Compute the distance of each embedding to each prototype."""

    expanded_prototypes = expand_dims(prototypes, 1)
    return sqrt(reduce_sum(
        square(expanded_prototypes - embeddings),
        2
    ))
