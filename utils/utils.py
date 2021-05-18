import logging
import os
import shutil
from os import DirEntry
from pathlib import Path
from typing import Dict, List


def make_dirs(
        directory: str
) -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def make_dir(
        base_dir: str,
        child_dir_name: str
) -> str:
    new_dir = os.path.join(base_dir, child_dir_name)
    make_dirs(new_dir)
    return new_dir


def copy_files_to_dir(files, directory):
    for file in files:
        copy_file_to_dir(file, directory)


def copy_file_to_dir(file, directory):
    logging.debug("Copy %s to %s", file, directory)
    shutil.copy2(file, directory)


def get_examples_by_category(
        directory: str
) -> Dict[DirEntry, List[str]]:
    with os.scandir(directory) as categories:
        return {
            category: [example.path for example in os.scandir(category) if example.is_file()]
            for category in categories
            if category.is_dir()
        }
