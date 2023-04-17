import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class BertSentenceEmbeddings():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_cosine_similarity(self, sent):
        marked_text = "[CLS] " + sent.text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding.detach().cpu().numpy() if torch.cuda.is_available() else sentence_embedding
    
    @staticmethod
    def find_most_similar_document_cosine(sentence_df, data_point) -> float:
        sentence_df["similarity_score"] = sentence_df["sentence_processed"].apply(lambda x: F.cosine_similarity(x, data_point, dim=0).item())
        return sentence_df.iloc[sentence_df["similarity_score"].idxmax()].sentence, sentence_df.iloc[sentence_df["similarity_score"].idxmax()].similarity_score