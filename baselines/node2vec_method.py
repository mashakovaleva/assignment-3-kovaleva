import numpy as np
import networkx as nx
from node2vec import Node2Vec
from ruwordnet.ruwordnet_reader import RuWordnet

def build_graph(ruwordnet, pos="N", directed=False):
    G = nx.DiGraph() if directed else nx.Graph()

    synsets = ruwordnet.get_all_synsets(pos)  # likely list of tuples (id, name)
    for row in synsets:
        synset_id = row[0] if isinstance(row, tuple) else row  # support both formats
        G.add_node(synset_id)

        hypers = ruwordnet.get_hypernyms_by_id(synset_id)  # list of synset IDs
        for h_id in hypers:
            G.add_edge(synset_id, h_id)

    return G

ruwordnet = RuWordnet(db_path="../data/ruwordnet.db", ruwordnet_path=None)
G = build_graph(ruwordnet, pos="N", directed=False) 
print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
node2vec = Node2Vec(G, dimensions=300, walk_length=20, num_walks=3, workers=1)
node2vec_model = node2vec.fit(window=10, min_count=1, epochs=7)
print('Saving...')
node2vec_model.wv.save_word2vec_format("models/vectors/node2vec/node2vec_ru_nouns_new.txt")
node2vec_model.save("models/node2vec_en_nouns_new")

def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis)) # (N,)
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for id, name in ruwordnet.get_all_synsets('N'):
        if ' ' not in name:
            word_embedding = source_dictionary[id]
            node_embedding = target_dictionary[id]
            source_matrix.append(word_embedding)
            target_matrix.append(node_embedding )
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, Vt = np.linalg.svd(product)
    return np.matmul(U, Vt)
