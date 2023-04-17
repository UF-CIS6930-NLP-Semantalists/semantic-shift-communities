import pandas as pd
import spacy
from spacy.tokens.token import Token
import string
import collections
from typing import List, Set


class Preprocessing():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def convert_to_string(self, data) -> str:
        return data if isinstance(data, str) else str(data)

    def tokenize(self, text: str) -> List[Token]:
        doc = self.nlp(text)
        return [w for sent in doc.sents for w in sent]

    def remove_punctuation(self, tokens: List[Token]) -> List[Token]:
        return [t for t in tokens if t.text not in string.punctuation]

    def remove_stop_words(self, tokens: List[Token]) -> List[Token]:
        return [t for t in tokens if not t.is_stop]

    def lemmatize(self, tokens: List[Token]) -> List[str]:
        return [t.lemma_ for t in tokens]

    def case_fold(self, tokens: List[str]) -> List[str]:
        return [t.lower() for t in tokens]

    def pre_process_text(self, text: str) -> List[str]:
        return self.case_fold(self.lemmatize(self.remove_punctuation(self.tokenize(self.convert_to_string(text)))))

    def get_num_sentences(self, data) -> int:
        doc = self.nlp(data if isinstance(data, str) else str(data))
        return len(list(doc.sents))

    def get_num_words(self, data) -> int:
        doc = self.nlp(data if isinstance(data, str) else str(data))
        sentence_length = 0
        for sent in doc.sents: sentence_length += len(sent)
        return sentence_length
    
    def get_processed_document(self, body):
        return self.nlp(body)
    
    def create_vocab(self, data: List[List[str]]) -> Set[str]:
        data = list(map(lambda x: x.strip('[]').split(",") if type(x) == str else [], data))
        vocab = {token.strip('" ') for tokens in data for token in tokens}
        return vocab
    
    def find_common_vocab(self, vocab1, vocab2):
        return list(vocab1.intersection(vocab2))


    def get_meta_data(self, dataframe):
        word_to_doc_map = collections.defaultdict(set)
        word_to_score_map = collections.defaultdict(int)
        word_to_freq_map = collections.defaultdict(int)

        for index, row in dataframe.iterrows():
            doc_id, score = row["id"], row["score"]
            try:
                for word in list(map(lambda x: x.strip("' \n"),row["preprocess_body"].strip('[]').split(','))):
                    word_to_doc_map[word].add(doc_id)
                    word_to_score_map[word] += score
                    word_to_freq_map[word] += 1
            except Exception as e:
                print("The row might be a non string value: ", e)

        meta_data = {"word": [], "freq": [], "score": [], "doc_ids": []}
        for word, freq in word_to_freq_map.items():
            meta_data["word"].append(word)
            meta_data["freq"].append(freq)
            meta_data["score"].append(word_to_score_map.get(word, 0))
            meta_data["doc_ids"].append(word_to_doc_map.get(word, []))

        return pd.DataFrame(meta_data)