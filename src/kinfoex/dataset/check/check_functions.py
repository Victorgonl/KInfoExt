import os
import json
from pprint import pprint
import re


def check_spaces(text):
    pattern = r"^\s+|\s+$"
    match = re.search(pattern, text)
    if match:
        return True
    else:
        return False


def check_texts(dataset):
    print("Checking texts:")
    bad_texts = {}
    for sample in dataset.values():
        entities = sample["data"]
        for entity in entities.values():
            if check_spaces(entity["text"]) or not entity["text"]:
                bad_texts[(sample["id"], entity["id"])] = entity["text"]
    if bad_texts:
        pprint(bad_texts, compact=True)
        return False
    print("OK!")
    return True


def check_duplications(dataset):
    print("Checking duplications:")
    duplications = {}
    for sample in dataset.values():
        entities = sample["data"]
        for entity_a in entities.values():
            links = []
            duplicated_links = []
            for link in entity_a["links"]:
                if link not in links:
                    links.append(link)
                else:
                    duplicated_links.append(link)
            if duplicated_links:
                duplication = (sample["id"], entity_a["id"])
                duplications[duplication] = duplicated_links
            for entity_b in entities.values():
                if entity_a["id"] != entity_b["id"]:
                    if (
                        entity_a["text"] == entity_b["text"]
                        and entity_a["box"] == entity_b["box"]
                    ):
                        duplication = tuple(sorted([entity_a["id"], entity_b["id"]]))
                        duplications[(sample["id"], duplication)] = entity_a["text"]
    if duplications:
        pprint(duplications, compact=True)
        return False
    print("OK!")
    return True


def check_labels(dataset):
    print("Checking labels:")
    LABELS = ["OTHER", "HEADER", "QUESTION", "ANSWER"]
    entidades_sem_label = {}
    for sample in dataset.values():
        entities = sample["data"]
        for entity in entities.values():
            if entity["label"] not in LABELS:
                if sample["id"] not in entidades_sem_label.keys():
                    entidades_sem_label[sample["id"]] = []
                entidades_sem_label[sample["id"]].append(f"{entity['id']}")
    if entidades_sem_label:
        pprint(entidades_sem_label, compact=True)
        return False
    print("OK!")
    return True


def check_links(dataset):
    print("Checking links:")
    failed_entities = {}
    for sample in dataset.values():
        entitie_type = {}
        sample_links = []
        entities = sample["data"]
        for entity in entities.values():
            entitie_type[entity["id"]] = entity["label"]
            for link in entity["links"]:
                if link[0] == entity["id"]:
                    sample_links.append(link)
            if entity["label"] == "ANSWER":
                if not entity["links"]:
                    failed_entities[(sample["id"], entity["id"]), entity["label"]] = (
                        entity["text"]
                    )
        total_of_qa_links = total_of_hq_links = total_of_hh_links = (
            total_of_ha_links
        ) = 0
        for link in sample_links:
            if (
                entitie_type[link[0]] == "QUESTION"
                and entitie_type[link[1]] == "ANSWER"
            ):
                total_of_qa_links += 1
            elif (
                entitie_type[link[0]] == "HEADER"
                and entitie_type[link[1]] == "QUESTION"
            ):
                total_of_hq_links += 1
            elif (
                entitie_type[link[0]] == "HEADER" and entitie_type[link[1]] == "HEADER"
            ):
                total_of_hh_links += 1
            elif (
                entitie_type[link[0]] == "HEADER" and entitie_type[link[1]] == "ANSWER"
            ):
                total_of_ha_links += 1
            else:
                if sample["id"] not in failed_entities.keys():
                    failed_entities[sample["id"]] = []
                failed_entities[sample["id"]].append(f"{link[0]}->{link[1]}")
    if failed_entities:
        pprint(failed_entities)
        return False
    print("OK!")
    return True
