from .check_functions import *


def check_kinfoext_dataset(kinfoext_dataset):

    print("Checking KInfoExt dataset...")
    print()
    labels_ok = check_labels(kinfoext_dataset)
    print()
    links_ok = check_links(kinfoext_dataset)
    print()
    duplications_ok = check_duplications(kinfoext_dataset)
    print()
    texts_ok = check_texts(kinfoext_dataset)

    return all([labels_ok, links_ok, duplications_ok, texts_ok])
