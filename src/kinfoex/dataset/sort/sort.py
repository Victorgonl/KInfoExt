import os
import json

from .json_encoder import UFLAFORMSJSONEncoder
from .sort_functions import *


def load_samples_data(data_folder):
    samples_data = {}
    for data_file in sorted(os.listdir(data_folder)):
        sample_id = data_file.split(".json")[0]
        data_path = os.path.join(data_folder, data_file)
        if os.path.isfile(data_path):
            data = json.load(open(data_path))
            samples_data[sample_id] = data
    return samples_data


def save_samples_data(data_folder, samples_data):
    for sample_id in samples_data:
        with open(f"{data_folder}/{sample_id}.json", "w") as f:
            json.dump(samples_data[sample_id], f, cls=UFLAFORMSJSONEncoder, indent=4)


def sort_uflaforms_dataset(uflaforms_dataset_directory, sort_by_relations=False):

    data_folder = "data"

    print("Sorting UFLA-FORMS dataset...")

    samples_data = load_samples_data(
        data_folder=f"{uflaforms_dataset_directory}/{data_folder}"
    )
    samples_data = sort_samples_data(
        samples_data=samples_data, sort_by_relations=sort_by_relations
    )
    save_samples_data(
        data_folder=f"{uflaforms_dataset_directory}/{data_folder}",
        samples_data=samples_data,
    )

    print()
    print("========================")
    print("|| UFLA-FORMS sorted! ||")
    print("========================")
    print()
