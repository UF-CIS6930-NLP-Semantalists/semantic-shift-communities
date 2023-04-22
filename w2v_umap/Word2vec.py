'''
requirements 
pip install gensim
pip install python-Levenshtein
'''

from gensim.models import Word2Vec
import pandas as pd
from matplotlib import pyplot as plt
import pickle


def body_only(df):
    return df['preprocess_body']

def clean_token_only(df):
    #[list(map(lambda x: x.strip("' \n"), body.strip('[]').split(','))) for body in df["preprocess_body"] if type(body) != float]
    return [list(map(lambda x: x.strip("' \n"), body.strip('[]').split(','))) for body in df if type(body) != float]

def tokens_extract(token_list):
    return [token[0] for token in token_list]

def sim_words(model, keyword, topn):
    return model.wv.most_similar(keyword, topn)

def word2vec_model_gen(input_tokens, vector_size, model_name):
    model = Word2Vec(input_tokens, vector_size = vector_size, window = 5, min_count =100, workers = 3, sg = 0)
    model.save(model_name)
    return "Saved model"
    
    
def model_load(model_path):
    return Word2Vec.load(model_path)


def get_exclusive_list(list1, list2):
    try: 
        list1_exclu = [word for word in list1 if word not in list2][:20]
        list2_exclu = [word for word in list2 if word not in list1][:20]
    
    except:
        print("too many words are overlapped")
        
    return list1_exclu, list2_exclu

def save_pkl(word_list, svg_path):
    with open(svg_path, "wb") as f: 
        pickle.dump(word_list, f)
    return
def load_pkl(svg_path):
    with open(svg_path, "rb") as f :
        data = pickle.load(f)
    return data


if __name__ == "__main__" :

	liberal_csv = pd.read_csv("liberal_base_sample.csv", header=0)
	conservative_csv = pd.read_csv("conservative_base_sample.csv", header = 0)

	liberal_body = body_only(liberal_csv)
	conservative_body = body_only(conservative_csv)

	liberal_body_tokens = clean_token_only(liberal_body)
	conservative_body_tokens = clean_token_only(conservative_body)

	save_pkl(liberal_body_tokens, "save_words_pickle/liberal_tokens_only")
	save_pkl(conservative_body_tokens, "save_words_pickle/conservative_tokens_only")

	#Model creation
	lib_model = word2vec_model_gen(liberal_body_tokens, 300,"w2v_liberal.model")
	con_model = word2vec_model_gen(conservative_body_tokens, 300, "w2v_conservative.model")

	lib_model.save("word2vec_lib.model")
	con_model.save("word2vec_con.model")

