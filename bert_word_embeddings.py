import torch
import numpy as np
from transformers import BertTokenizer, AutoModelForMaskedLM
import pickle
from collections import defaultdict

class BertWordEmbeddings():
    def __init__(self, model_checkpoint):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint,
                                                          output_hidden_states=True,
                                                          use_auth_token='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')
        self.model.to(self.device)
        self.model.eval()
        
    def generate_encodings(self, reddit_posts):
        # Encode the tokens
        encoded_posts = [self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        ) for post in reddit_posts]
        encoded_posts = [encoded_post.to(self.device) for encoded_post in encoded_posts] if torch.cuda.is_available() else encoded_posts
        return encoded_posts

    def runFFN(self, encoded_posts):
        word_vector = {}
        word_freq = defaultdict(int)

        for post in encoded_posts:
            with torch.no_grad():
                output = self.model(**post)
                hidden_states = output.hidden_states
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = token_embeddings.permute(1, 2, 0, 3)
                word_embeddings = torch.mean(token_embeddings, dim=2)
                tokens = [self.tokenizer.convert_ids_to_tokens(post['input_ids'][0])]
                for i in range(len(word_embeddings)):
                    for j in range(len(word_embeddings[i])):
                        if tokens[i][j] in ['[CLS]', '[SEP]', '[PAD]']: continue
                        if tokens[i][j] in word_vector:word_vector[tokens[i][j]] = word_vector[tokens[i][j]] + word_embeddings[i][j]
                        else: word_vector[tokens[i][j]] = word_embeddings[i][j]
                        word_freq[tokens[i][j]] += 1

        for token in word_vector:
            word_vector[token] = (word_vector[token] / word_freq[token]).detach().cpu().numpy()
        
        return word_vector, word_freq

    @staticmethod
    def save_pkl(data_structure, svg_path):
        with open(svg_path, "wb") as f: 
            pickle.dump(data_structure, f)

    @staticmethod
    def load_pkl(svg_path):
        with open(svg_path, "rb") as f :
            data = pickle.load(f)
        return data