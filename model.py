import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_cos_sim

from torch.nn import CrossEntropyLoss
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, args, tokenizer, encode_style="emarker"):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased').to(args.device)
        self.model.resize_token_embeddings(len(tokenizer))

        # 'cls' using the cls_token as the embedding
        # 'emarker' concatenating the embedding of head and tail entity markers
        if encode_style in ["cls", "emarker"]:
            self.encode_style = encode_style
        else:
            raise Exception("Encode_style must be 'cls' or 'emarker'.")

        if encode_style == "emarker":
            hidden_size = self.model.config.hidden_size
            self.linear_transform = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size, bias=True),
                nn.GELU(),
                nn.LayerNorm([hidden_size])
            ).to(self.args.device)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.args.feature_dim)
            ).to(self.args.device)


    def forward(self, input_ids, attention_mask, h_index, t_index, labels = None):
        plm_output = self.model(input_ids, attention_mask=attention_mask)['last_hidden_state']
        if self.encode_style == "cls":
            hidden = plm_output.index_select(1, torch.tensor([0]).to(self.args.device)).squeeze()  # [batch_size, hidden_size]
        else:
            h = torch.stack([plm_output[i, h_index[i], :] for i in range(len(h_index))], dim=0) # [batch_size, hidden_size]
            t = torch.stack([plm_output[i, t_index[i], :] for i in range(len(t_index))], dim=0) # [batch_size, hidden_size]
            ht_embeddings = torch.cat([h, t], dim=1) # [batch_size, hidden_size*2]
            hidden = self.linear_transform(ht_embeddings) # [batch_size, hidden_size]
            feature = self.head(hidden) # [batch_size, feature_dim]
            feature = F.normalize(feature, p=2, dim=1) # [batch_size, feature_dim]

        output = (hidden, feature)
        
        if labels is not None:
            # compute scloss of current task
            dot_div_temp = torch.mm(feature, feature.T) / self.args.cl_temp # [batch_size, batch_size]
            dot_div_temp_norm = dot_div_temp - torch.max(dot_div_temp, dim=1, keepdim=True)[0].detach() # [batch_size, batch_size]
            exp_dot_temp = torch.exp(dot_div_temp_norm) + 1e-8 # avoid log(0)  [batch_size, batch_size]

            mask = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels).to(self.args.device) # [batch_size, batch_size]
            cardinalities = torch.sum(mask, dim=1) # [batch_size]

            log_prob = -torch.log(exp_dot_temp / torch.sum(exp_dot_temp, dim=1, keepdim=True)) # [batch_size, batch_size]
            scloss_per_sample = torch.sum(log_prob*mask, dim=1) / cardinalities # [batch_size]
            scloss = torch.mean(scloss_per_sample)

            loss = scloss
            output = (loss, ) + output

        return output
    
    def get_low_dim_feature(self, hidden):
        feature = self.head(hidden)
        feature = F.normalize(feature, p=2, dim=1)
        return feature


class Classifier(nn.Module):
    def __init__(self, args, hidden_dim, label_num, prev_classifier=None):
        super().__init__()
        self.args = args
        self.label_num = label_num
        self.classifier = nn.Linear(hidden_dim, label_num, bias=False)
        self.loss_fn = CrossEntropyLoss()
    
    def forward(self, hidden, labels=None):
        logits = self.classifier(hidden)
        output = (logits, )

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output = (loss, ) + output

        return output
    
    def incremental_learning(self, seen_rel_num):
        weight = self.classifier.weight.data
        self.classifier = nn.Linear(768, seen_rel_num, bias=False).to(self.args.device)
        with torch.no_grad():
            self.classifier.weight.data[:seen_rel_num] = weight[:seen_rel_num]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
