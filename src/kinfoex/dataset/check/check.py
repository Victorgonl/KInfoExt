from .check_functions import *


def check_uflaforms_dataset(uflaforms_dataset):

    print("Checking UFLA-FORMS dataset...")
    print()
    labels_ok = check_labels(uflaforms_dataset)
    print()
    links_ok = check_links(uflaforms_dataset)
    print()
    duplications_ok = check_duplications(uflaforms_dataset)
    print()
    texts_ok = check_texts(uflaforms_dataset)

    return all([labels_ok, links_ok, duplications_ok, texts_ok])
