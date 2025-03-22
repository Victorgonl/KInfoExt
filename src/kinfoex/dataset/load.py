import os
import glob
import json

import PIL.Image

DATA_FOLDER = "data"
IMAGE_FOLDER = "image"

DATA_FILE_EXTENSION = "json"
IMAGE_FILE_EXTENTIONS = ["jpg", "png"] + [
    extension.replace(".", "") for extension in PIL.Image.registered_extensions()
]

DATASET_INFO_FILE = "dataset_info.json"


def load_kinfoex_dataset(dataset_directory, load_images=True, split_option=None):
    kinfoex_dataset = {}
    image_folder = f"{dataset_directory}/{IMAGE_FOLDER}"
    data_folder = f"{dataset_directory}/{DATA_FOLDER}"

    dataset_info_file = f"{dataset_directory}/{DATASET_INFO_FILE}"
    if os.path.exists(dataset_info_file):
        with open(dataset_info_file) as dataset_info_file:
            dataset_info = json.load(dataset_info_file)
            labels = dataset_info["labels"]
            relations = dataset_info["relations"]
            if split_option is None:
                split_options = dataset_info["splits"]
            else:
                split_options = [split_option]
            for split_option in split_options:
                for split in dataset_info["splits"][split_option]:
                    for sample_id in dataset_info["splits"][split_option][split]:
                        kinfoex_dataset[sample_id] = {}
                        kinfoex_dataset[sample_id]["split"] = split
                        kinfoex_dataset[sample_id]["labels"] = labels
                        kinfoex_dataset[sample_id]["relations"] = relations
    else:
        data_files = glob.glob(os.path.join(data_folder, f"*.{DATA_FILE_EXTENSION}"))
        samples_ids = [
            os.path.splitext(os.path.basename(file))[0] for file in data_files
        ]
        for sample_id in samples_ids:
            kinfoex_dataset[sample_id] = {}
            kinfoex_dataset[sample_id]["split"] = None
            kinfoex_dataset[sample_id]["labels"] = None
            kinfoex_dataset[sample_id]["relations"] = None

    images = []
    datas = []
    for sample_id in kinfoex_dataset:
        # load data
        data = None
        data_path = f"{data_folder}/{sample_id}.{DATA_FILE_EXTENSION}"
        if os.path.isfile(data_path):
            data = json.load(open(data_path))
            data_mapped = {}
            for entity in data:
                data_mapped[entity["id"]] = entity
            datas.append(data_mapped)
        # load image
        image = None
        if load_images:
            for image_extension in IMAGE_FILE_EXTENTIONS:
                image_file = f"{sample_id}.{image_extension}"
                image_path = os.path.join(image_folder, image_file)
                if os.path.isfile(image_path):
                    image = PIL.Image.open(image_path)
        images.append(image)

    for sample_id, image, data in zip(list(kinfoex_dataset.keys()), images, datas):
        sample = {
            "id": sample_id,
            "split": None,
            "labels": None,
            "relations": None,
            "image": image,
            "data": data,
        }
        kinfoex_dataset[sample_id] = sample

    return kinfoex_dataset
