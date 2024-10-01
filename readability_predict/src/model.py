import torch
from transformers import BertModel, AutoModel
from torch import nn
import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=5):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class BertForSeq(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.num_labels = args.num_labels
        self.bert = AutoModel.from_pretrained(args.model_path)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.linear_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, args.hidden_size),
            nn.Dropout(args.dropout),
        )

        self.bert_head = nn.Sequential(
            nn.Linear(args.hidden_size, args.num_labels),
        )

        self.bert_feature_head = nn.Sequential(
            nn.Linear(args.hidden_size + args.num_feature, args.num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, feature=None, labels=None):
        loss = None
        if not feature:
            output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = self.bert_head(self.linear_head(output[1]))
        else:
            output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = self.bert_feature_head(torch.cat((self.linear_head(output[1]), feature), 1))
        if labels is not None:
            # loss_fc = nn.CrossEntropyLoss()
            loss_fc = SCELoss(alpha=0.1, beta=1, num_classes=self.num_labels)
            loss = loss_fc(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits




