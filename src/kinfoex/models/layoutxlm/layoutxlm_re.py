from transformers import LayoutLMv2Model, LayoutLMv2PreTrainedModel

from torch import nn

from ..re_decoder import RelationExtractionOutput, RelationExtractionDecoder


class LayoutXLMForRelationExtraction(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = RelationExtractionDecoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        labels=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        entities=None,
        relations=None,
    ):
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        seq_length = input_ids.size(1)
        sequence_output, image_output = (
            outputs[0][:, :seq_length],
            outputs[0][:, seq_length:],
        )
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return RelationExtractionOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )
