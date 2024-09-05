import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch.nn import functional as F

from config import HIDDEN_DIM, MLP_DIM

class BertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim = HIDDEN_DIM, dropout=0.1):
        super(BertMultiClassifier, self).__init__()


        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'dropout': dropout
        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, labels_count)
        self.sigmoid = nn.Sigmoid()


    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks)
        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)

        return proba







class ExtraBertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim = HIDDEN_DIM, mlp_dim =MLP_DIM, extras_dim = 6 ,dropout=0.1):
        super(ExtraBertMultiClassifier, self).__init__()


        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'mlp_dim':mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout

        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        


        self.softmax  = nn.Softmax()
    
    def forward(self, tokens, masks, extras):
        '''
            tokens: [batch_size, seq_len, 768]
            masks: [batch_size, seq_len, 1]
            extras: [batch_size, seq_lem, extras_dim]
        '''
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)

        concat_output = torch.cat([dropout_output, extras], dim=1)

        mlp_output = self.mlp(concat_output)
        proba = self.softmax(mlp_output)

        return proba







class LinearMultiClassifier(nn.Module):
    pass






class ExtraMultiClassifier(nn.Module):
    pass