import sys
import json
import codecs

from predict_models import BaselineModel, SecondOrderModel, SecondOrderModelTransform


def save_to_file(words_with_hypernyms, output_path, ruwordnet):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            for hypernym in hypernyms:
                f.write(f"{word}\t{hypernym}\t{ruwordnet.get_name_by_id(hypernym)}\n")


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def main():
    models = {"baseline": BaselineModel, "second_order": SecondOrderModel, 'second_order_transform': SecondOrderModelTransform}
    params = load_config()
    
    test_data = []
    with open(params['test_path'], 'r', encoding='utf-8') as f:
        #test_data = f.read().split("\n")[:-1]
        for line in f:
            line = line.strip()
            if not line:
                continue
            phrase = line.split('\t', 1)[0].strip().lower() # it can be both one new word and new words combination 
            phrase = phrase.upper().replace(" ", "_")
            if phrase:
                test_data.append(phrase)
    
    baseline = models[params["model"]](params)
    print("Model loaded")
    results = baseline.predict_hypernyms(list(test_data))
    save_to_file(results, params['output_path'], baseline.ruwordnet)


if __name__ == '__main__':
    main()
