import torch
import torch.nn as nn

class BertPreTrain(nn.Module):
    def __init__(self, bert_model):
        super(BertPreTrain, self).__init__()
        self.bert = bert_model
        self.mlm = MaskedLanguageModel(bert_model.config.vocab_size, bert_model.config.hidden_size)
        self.nsp = NextSentencePrediction(bert_model.config.hidden_size)

    def forward(self, x, mask, segment_ids, mlm_positions):
        x = self.bert(x, mask, segment_ids)
        prediction_scores = self.mlm(x, mlm_positions)
        next_sentence_scores = self.nsp(x)
        return prediction_scores, next_sentence_scores

class MaskedLanguageModel(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(MaskedLanguageModel, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mlm_positions):
        batch_size = mlm_positions.shape[0]
        return torch.tensor(
            [torch.log_softmax(
                self.fc(x), dim=-1).detach().numpy()[i, mlm_positions[i, :], :] for i in range(batch_size)])

class NextSentencePrediction(nn.Module):
    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        pooler_output = self.pooler(x[:, 0, :])
        return torch.log_softmax(self.fc(pooler_output), dim=-1)
