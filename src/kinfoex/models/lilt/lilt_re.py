from transformers import LiltModel, LiltPreTrainedModel

from torch import nn

from ..re_decoder import RelationExtractionOutput, RelationExtractionDecoder


class LiLTForRelationExtraction(LiltPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.lilt = LiltModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = RelationExtractionDecoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entities=None,
        relations=None,
    ):

        outputs = self.lilt(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return RelationExtractionOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
        )
