from rdkit import RDLogger

import tensorflow as tf
from tensorflow import keras
from simple_ggan import graph_to_molecule

RDLogger.DisableLog("rdApp.*")


def BaseGenerator(dense_units, dropout_rate, latent_dim,
                  adjacency_shape, feature_shape,):
    z = keras.layers.Input(shape=(latent_dim,))
    # Propagate through one or more densely connected layers
    x = z
    for units in dense_units:
        x = keras.layers.Dense(units, activation="tanh")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    # Map outputs of previous layer (x) to [continuous] adjacency
    #                                                   tensors (x_adjacency)
    x_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
    x_adjacency = keras.layers.Reshape(adjacency_shape)(x_adjacency)
    # Symmetrify tensors in the last two dimensions
    x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
    x_adjacency = keras.layers.Softmax(axis=1)(x_adjacency)

    # Map outputs of previous layer (x) to [continuous]
    #                                           feature tensors (x_features)
    x_features = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
    x_features = keras.layers.Reshape(feature_shape)(x_features)
    x_features = keras.layers.Softmax(axis=2)(x_features)

    return keras.Model(inputs=z, outputs=[x_adjacency, x_features],
                       name="Generator")


def sample_molecules(generator, batch_size, latent_dim, bond_dim, atom_dim,
                     atom_mapping, bond_mapping):
    z = tf.random.normal((batch_size, latent_dim))
    graph = generator.predict(z)

    # obtain one-hot encoded adjacency tensor
    adjacency = tf.argmax(graph[0], axis=1)
    adjacency = tf.one_hot(adjacency, depth=bond_dim, axis=1)

    # Remove potential self-loops from adjacency
    adjacency = tf.linalg.set_diag(adjacency,
                                   tf.zeros(tf.shape(adjacency)[:-1]))

    # obtain one-hot encoded feature tensor
    features = tf.argmax(graph[1], axis=2)
    features = tf.one_hot(features, depth=atom_dim, axis=2)

    return [graph_to_molecule([adjacency[i].numpy(), features[i].numpy()],
                              atom_dim, bond_dim, atom_mapping, bond_mapping)
            for i in range(batch_size)]
