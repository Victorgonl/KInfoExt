from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class UFLAFORMSDataCollator:

    pad_token_label = -100
    valid_keys = [
        "input_ids",
        "bbox",
        "labels",
        "attention_mask",
        "entities",
        "relations",
        "image",
    ]

    def __call__(self, features):

        if not isinstance(features, list):
            features = [features]

        keys = [key for key in list(features[0].keys()) if key in self.valid_keys]
        batch_lists = {key: [] for key in keys}
        for key in batch_lists:
            for feature in features:
                for i in range(len(feature["attention_mask"])):
                    if not feature["attention_mask"][i]:
                        feature["labels"][i] = self.pad_token_label
                batch_lists[key].append(feature[key])

        batch_tensors = {}
        for key in batch_lists:
            if key not in ["entities", "relations"]:
                batch_tensors[key] = torch.tensor(np.array(batch_lists[key]))
            else:
                batch_tensors[key] = batch_lists[key]

        return batch_tensors
