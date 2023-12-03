import os
import random
import string
from typing import List, Dict


def random_string_generator(k=4):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))


def write_data_to_file(
        document_dir: str = None, full_documents: List[Dict[str, str]] = None, single_file_flag: bool = False
):
    """
    Write various full documents from `full_documents` to document_dir.
    :param document_dir: "data/past_surveys/"
    :param full_documents: [{"file_name": "file_name", "full_document": "full_document"}]
    :param single_file_flag: Save as single file or multiple files flag.
    :return:
    """
    directory_name = '_+_'.join([doc['file_name'].replace(' ', "_")[:15] for doc in full_documents])
    directory_name = f"{directory_name}_{random_string_generator()}"
    if single_file_flag:
        full_directory_path = os.path.join(document_dir, directory_name)
        os.makedirs(full_directory_path, exist_ok=True)
        data_file_name = directory_name + ".data.txt"
        meta_file_name = directory_name + ".meta.txt"
        full_data_file_path = os.path.join(full_directory_path, data_file_name)
        full_meta_file_path = os.path.join(full_directory_path, meta_file_name)
        full_data_text = '\n\n\n'.join(
            f"""```\nFILE TITLE: {doc['file_name']}\nCONTENT: {doc['extracted_text']}\n```"""
            for doc in full_documents
        )
        full_meta_text = full_documents[0]['doc_description']
        with open(full_data_file_path, 'w') as file:
            file.write(full_data_text)
        with open(full_meta_file_path, 'w') as file:
            file.write(full_meta_text)
    else:
        file_names = [doc['file_name'].replace(' ', "_") for doc in full_documents]
        full_texts = [doc['extracted_text'] for doc in full_documents]
        full_meta_text = full_documents[0]['doc_description']

        # save meta
        meta_file_name = directory_name + ".meta.txt"
        full_directory_path = os.path.join(document_dir, directory_name)
        os.makedirs(full_directory_path, exist_ok=True)
        full_meta_file_path = os.path.join(full_directory_path, meta_file_name)
        with open(full_meta_file_path, 'w') as file:
            file.write(full_meta_text)

        for file_name, full_text in zip(file_names, full_texts):
            file_name = f"{file_name[:15]}_{random_string_generator()}.data.txt"
            full_directory_path = os.path.join(document_dir, directory_name)
            os.makedirs(full_directory_path, exist_ok=True)
            full_file_path = os.path.join(full_directory_path, file_name)
            with open(full_file_path, 'w') as file:
                file.write(full_text)
    return directory_name
