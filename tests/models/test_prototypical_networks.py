import tensorflow_fewshot.models.prototypical_network as pn
from numpy.random import normal

def test_l2_norm_shape_is_10_5_when_passed_10_prototypes_10_embeddings():
    # Given
    prototypes = normal(size=(5, 64))
    query_embeddings = normal(size=(10, 64))

    # When
    distances = pn._euclidean_distance(prototypes, query_embeddings)

    # Then
    assert distances.shape == (5, 10)

def test_encoder_output_shape_is_10_64_when_passed_10_images():
    # Given
    encoder = pn._create_imageNetCNN()
    images = normal(size=(10,28,28,1))

    # When
    embeddings = encoder(images)

    # Then
    assert embeddings.shape == (10, 64)
