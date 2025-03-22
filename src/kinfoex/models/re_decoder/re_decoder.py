import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.file_utils import ModelOutput


from ..biaffine_attention import BiaffineAttention


class RelationExtractionDecoder(nn.Module):

    NEGATIVE_RELATION = 0

    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(
            config.hidden_size // 2, config.num_labels
        )
        self.loss_fct = CrossEntropyLoss()

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (
                entities["start"][rel["head_id"]],
                entities["end"][rel["head_id"]],
            )
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (
                entities["start"][rel["tail_id"]],
                entities["end"][rel["tail_id"]],
            )
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            label = pred_label.data.tolist()
            rel["type"] = label
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(
                logits, relations[b], entities[b]
            )
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None  # type: ignore
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None
