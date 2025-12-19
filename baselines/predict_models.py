from abc import abstractmethod
from collections import defaultdict

from ruwordnet.ruwordnet_reader import RuWordnet
from gensim.models import KeyedVectors
import numpy as np


class Model:
    def __init__(self, params):
        self.ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
        self.w2v_ruwordnet = KeyedVectors.load_word2vec_format(params['ruwordnet_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    @abstractmethod
    def predict_hypernyms(self, neologisms, topn=10):
        pass

    @abstractmethod
    def __compute_hypernyms(self, neologisms, topn=10):
        pass


class BaselineModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}

    def __compute_hypernyms(self, neologism, topn=10) -> list:
        return [i[0] for i in self.w2v_ruwordnet.similar_by_vector(self.w2v_data[neologism], topn)]


class SecondOrderModelTransform(Model):
    def __init__(self, params):
        super().__init__(params)
        self.transform = np.load('ruwordnet_ft_to_n2v_transform_new.npy')

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}

    def __compute_hypernyms(self, neologism, topn=10, k_assoc=25):
        vec = self.w2v_data[neologism]
        vec = vec @ self.transform
        associates = self.w2v_ruwordnet.similar_by_vector(vec, topn=k_assoc)
        hypernym_scores = defaultdict(float)
        for synset_id, sim in associates:
            for h in self.ruwordnet.get_hypernyms_by_id(synset_id):
                if h.endswith("-N"): 
                    hypernym_scores[h] += float(sim)

        ranked = sorted(hypernym_scores.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:topn]]
    
class SecondOrderModel(Model):  
    def __init__(self, params):
        super().__init__(params)

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}
   
    def __compute_hypernyms(self, neologism, topn=10, k_assoc=100):
        associates = self.w2v_ruwordnet.similar_by_vector(
            self.w2v_data[neologism], topn=k_assoc
        )
        hypernym_scores = defaultdict(float)
        for synset_id, sim in associates:
            for h in self.ruwordnet.get_hypernyms_by_id(synset_id): 
                hypernym_scores[h] += float(sim)

        ranked = sorted(
            hypernym_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [h for h, _ in ranked[:topn]]
    
