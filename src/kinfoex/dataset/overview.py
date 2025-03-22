from datasets import concatenate_datasets

RELATION_SYMBOL = "->"
NULL_RELATION_ID = 0
NULL_TOKEN_ID = 0


def get_uflaforms_overview(uflaforms_dataset):
    overview = {
        "samples": {"total": 0},
        "words": {"total": 0, "total_entities": 0},
        "entities": {"total": 0, "total_entities": 0},
        "relations": {"total": 0},
    }
    for sample_id in uflaforms_dataset:
        data = uflaforms_dataset[sample_id]["data"]
        labels = uflaforms_dataset[sample_id]["labels"]
        overview["samples"]["total"] += 1
        for entity_id in data:
            entity = data[entity_id]

            label = entity["label"]
            overview["entities"]["total"] += 1
            if label not in overview["words"]:
                overview["entities"][label] = 0
            overview["entities"][label] += 1

            words = entity["words"]
            overview["words"]["total"] += len(words)
            if label not in overview["words"]:
                overview["words"][label] = 0
            overview["words"][label] += len(words)

            if labels is not None and label in labels:
                overview["entities"]["total_entities"] += 1
                overview["words"]["total_entities"] += len(words)

            links = entity["links"]
            for link in links:
                k, v = link[0], link[1]
                if k == entity_id:
                    relation_label = (
                        f"{data[k]['label']}{RELATION_SYMBOL}{data[v]['label']}"
                    )
                    overview["relations"]["total"] += 1
                    if relation_label not in overview["relations"]:
                        overview["relations"][relation_label] = 0
                    overview["relations"][relation_label] += 1
    return overview


def get_processed_uflaforms_overview(uflaforms_dataset):
    overview = {
        "samples": {"total": 0},
        "tokens": {"total": 0},
        "entities_tokens": {"total": 0},
        "entities": {"total": 0},
        "relations": {"total": 0},
    }
    uflaforms_dataset = concatenate_datasets(
        [uflaforms_dataset[partition] for partition in uflaforms_dataset]
    )
    tokens_id2label = {
        i: label
        for i, label in enumerate(uflaforms_dataset.features["labels"].feature.names)
    }
    entities_id2label = {
        i: label
        for i, label in enumerate(
            uflaforms_dataset.features["entities"].feature["label"].names
        )
    }
    relations_id2label = {
        i: label
        for i, label in enumerate(
            uflaforms_dataset.features["relations"].feature["label"].names
        )
    }

    for sample in uflaforms_dataset:

        overview["samples"]["total"] += 1

        for i in range(len(sample["input_ids"])):
            if sample["attention_mask"][i]:
                overview["tokens"]["total"] += 1
                label = tokens_id2label[sample["labels"][i]]
                if label not in overview["tokens"]:
                    overview["tokens"][label] = 0
                overview["tokens"][label] += 1

        entities = sample["entities"]
        for start, end, label in zip(
            entities["start"], entities["end"], entities["label"]
        ):

            # this check if the entity is a valid one
            token_label_sequence = sample["labels"][start:end]
            if NULL_TOKEN_ID in token_label_sequence:
                continue

            label = entities_id2label[label]
            overview["entities"]["total"] += 1
            if label not in overview["entities"]:
                overview["entities"][label] = 0
            overview["entities"][label] += 1

            n_tokens = end - start
            overview["entities_tokens"]["total"] += n_tokens
            if label not in overview["entities_tokens"]:
                overview["entities_tokens"][label] = 0
            overview["entities_tokens"][label] += n_tokens

        relations = sample["relations"]
        for head, tail, label in zip(
            relations["head"], relations["tail"], relations["label"]
        ):
            if label != NULL_RELATION_ID:
                label = relations_id2label[label]
                overview["relations"]["total"] += 1
                if label not in overview["relations"]:
                    overview["relations"][label] = 0
                overview["relations"][label] += 1

    for label in entities_id2label.values():
        assert overview["entities_tokens"][label] == sum(
            [
                overview["entities_tokens"][l]
                for l in overview["entities_tokens"]
                if label in l
            ]
        )

    return overview
