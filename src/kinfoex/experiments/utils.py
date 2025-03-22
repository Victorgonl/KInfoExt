import random
import math


def random_split_dataset(dataset, ratio, seed=None):
    if seed is not None:
        random.seed(seed)
    samples_ids = set(dataset["original_id"])
    num_to_select = math.floor(len(samples_ids) * ratio)
    selected_ids = random.sample(sorted(samples_ids), num_to_select)
    selected, not_selected = set(), set()
    for i, sample in enumerate(dataset):
        if sample["original_id"] in selected_ids:
            selected.add(i)
        else:
            not_selected.add(i)
    selected, not_selected = sorted(selected), sorted(not_selected)
    return dataset.select(selected), dataset.select(not_selected)
