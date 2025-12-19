from collections import defaultdict
import argparse

import numpy as np
from gensim.models.fasttext import load_facebook_model
from string import punctuation
from ruwordnet.ruwordnet_reader import RuWordnet

from gpt_dicts.public_nouns import enrich_public_nous, normalize_key
#from gpt_dicts.private_nouns import enrich_private_nous


class FasttextVectorizer:
    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)
        print('Model loaded')

    # -------------------------------------------------------------
    # vectorize ruwordnet
    # -------------------------------------------------------------

    def vectorize_ruwordnet(self, synsets, output_path):
        ids, vectors = self.__get_ruwordnet_vectors(synsets)
        self.save_as_w2v(ids, vectors, output_path)

    def __get_ruwordnet_vectors(self, synsets):
        ids = []
        vectors = np.zeros((len(synsets), self.model.vector_size))
        for i, (_id, texts) in enumerate(synsets.items()):
            ids.append(_id)
            vectors[i, :] = self.__get_avg_vector(texts)
        return ids, vectors

    def __get_avg_vector(self, texts):
        sum_vector = np.zeros(self.model.vector_size)
        for text in texts:
            words = [i.strip(punctuation) for i in text.split()]
            sum_vector += np.sum(self.__get_data_vectors(words), axis=0)/len(words)
        return sum_vector/len(texts)

    # -------------------------------------------------------------
    # vectorize data
    # -------------------------------------------------------------

    def vectorize_data(self, data, output_path, save_first_word=False):
        data_vectors = self.__get_data_vectors(data)
        self.save_as_w2v(data, data_vectors, output_path, save_first_word)

    def __get_data_vectors(self, data):
        vectors = np.zeros((len(data), self.model.vector_size))
        for i, phrase in enumerate(data):  # TODO: how to do it more effective or one-line
            words = phrase.split()
            vecs = [self.model.wv[w] for w in words]
            vectors[i, :] = np.mean(vecs, axis=0)
        return vectors

    # -------------------------------------------------------------
    # save
    # -------------------------------------------------------------

    @staticmethod
    def save_as_w2v(phrases: list, vectors: np.array, output_path: str, save_first_word=False):
        assert len(phrases) == len(vectors)
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
            for phrase, vector in zip(phrases, vectors):
                if save_first_word:
                    words = phrase.split()
                    phrase = words[0].upper()
                else:
                    phrase = phrase.upper().replace(" ", "_")
                vector_line = " ".join(map(str, vector))
                w.write(f"{phrase} {vector_line}\n")

def process_data(input_file, output_file, ft_vec, with_dict=False):
    phrases = []
    with open(input_file, 'r', encoding='utf-8') as f:
        #dataset = f.read().lower().split("\n")[:-1]
        for line in f:
            line = line.strip()
            if not line:
                continue
            phrase = line.split('\t', 1)[0] # it can be both one new word and new words combination
            phrase = phrase.strip().lower() 
            if with_dict:
                #related_words = ' '.join(enrich_public_nous[normalize_key(phrase)]) 
                phrase = phrase + ' ' + enrich_public_nous[normalize_key(phrase)][0] # related_words
            if phrase:
                phrases.append(phrase)
    print(f'In process data {phrases[:2]}')
    ft_vec.vectorize_data(phrases, output_file, save_first_word=with_dict)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--use_gpt_dict',
        action='store_true',
        help='Enrich embeddings with GPT-generated synonyms/hypernyms'
    )
    args = p.parse_args()
    with_dict = args.use_gpt_dict
    ft_vec = FasttextVectorizer("models/cc.ru.300.bin")
    ruwordnet = RuWordnet(db_path="../data/ruwordnet.db", ruwordnet_path=None)
    noun_synsets = defaultdict(list)
    verb_synsets = defaultdict(list)
    for sense_id, synset_id, text in ruwordnet.get_all_senses():
        if synset_id.endswith("N"):
            noun_synsets[synset_id].append(text.lower())
        elif synset_id.endswith("V"):
            verb_synsets[synset_id].append(text.lower())

    ft_vec.vectorize_ruwordnet(noun_synsets, "models/vectors/ruwordnet_nouns_fasttext.txt")
    ft_vec.vectorize_ruwordnet(verb_synsets, "models/vectors/ruwordnet_verbs_fasttext.txt")

    enrich_public_nous = {normalize_key(k): v for k, v in enrich_public_nous.items()}
    # process_data("../data/public_test/verbs_public.tsv", "models/vectors/verbs_public_fasttext1.txt")
    #process_data("../data/public_test/nouns_public.tsv", "models/vectors/fasttext_with_gpt_dict/nouns_public_fasttext2.txt")
   # process_data("../data/public_test/verbs_public.tsv", "models/vectors/fasttext_with_gpt_dict/verbs_public_fasttext.txt")
    #process_data("../data/private_test/nouns_private.tsv", "models/vectors/fasttext_with_gpt_dict/nouns_private_fasttext.txt")
    process_data("../data/private_test/verbs_private.tsv", "models/vectors/fasttext_with_gpt_dict/verbs_private_fasttext.txt")
    # process_data("../data/private_test/nouns_private.tsv", "models/vectors/nouns_private_fasttext1.txt")
    # process_data("../data/training_data/all_data_nouns.tsv", "models/vectors/training_all_data_nouns_fasttext1.txt")
    # process_data("../data/training_data/all_data_verbs.tsv", "models/vectors/training_all_data_verbs_fasttext1.txt")
