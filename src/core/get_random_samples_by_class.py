from pathlib import Path
import random

from src import Config
from src.logs import logs
from src._T_typing import (
    _T_pointer2class_dataset,
    _T_pointerValues,
    _T_label,
    _T__constructor__,
)

def get_random_samples_by_class(
    source_root_path: Path,
    class_names: list[str],
    samples_amount_per_all_class: int,

    save_project_relative_path: Path = Path("data/test"),
) -> _T_pointer2class_dataset:
    """
        The function is responsible for selecting N random samples
        for each selected gesture class.

        The source directory should contain subdirectories representing
        individual sample classes. The names of these directories are
        treated as class labels.

        The function checks whether all requested classes exist and
        whether the required number of samples is available for each
        selected class. If the requested classes or number of samples
        are not available, an exception is raised.

        For every requested class, N files are randomly selected without
        replacement. The resulting information is stored in a dataset
        description structure and saved as a JSON file.

        Class labels must correspond to the names of the directories
        in the source dataset.

        :param source_root_path: Path to the root directory containing
            the source dataset.
        :param class_names: List of class names for which samples
            should be selected.
        :param samples_amount_per_all_class: Number of random samples
            to select for each class.
        :param save_project_relative_path: Project-relative path to
            the directory where the result should be saved.
        :param save_nametag: Optional suffix added to the generated
            output filename.
        :return: Dataset description containing the selected samples
            for each requested class.
    """
    logs.print("d", "get_random_samples_by_class")

    # ------------------------------------------------------------------
    # Validate input arguments
    # ------------------------------------------------------------------
    if not isinstance(source_root_path, Path):
        raise TypeError("Valid source_root_path type: Path")
    print(source_root_path)
    if not source_root_path.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source_root_path}"
        )

    if not source_root_path.is_dir():
        raise NotADirectoryError(
            f"Source path must be a directory: {source_root_path}"
        )

    if not isinstance(class_names, list):
        raise TypeError("Valid class_names type: must be list")

    if not class_names:
        raise ValueError("Valid length of class_names: must be > 0")

    if not all(isinstance(name, str) for name in class_names):
        raise TypeError(
            "Valid class_names type: all class names must be strings"
        )

    if not isinstance(samples_amount_per_all_class, int):
        raise TypeError(
            "Valid samples_amount_per_all_class type: must be int"
        )

    if samples_amount_per_all_class <= 0:
        raise ValueError(
            "Valid samples_amount_per_all_class: must be > 0"
        )

    # ------------------------------------------------------------------
    # Find available classes in the source dataset
    # ------------------------------------------------------------------
    available_class_paths = [
        path
        for path in source_root_path.iterdir()
        if path.is_dir()
    ]

    # Create a mapping:
    # class name -> (number of available entries, class directory)
    class_name_count_path: dict[str, tuple[int, Path]] = {
        class_path.name: (
            len(list(class_path.iterdir())),
            class_path,
        )
        for class_path in available_class_paths
    }

    logs.print(
        "i",
        (
            f"Found {len(available_class_paths)} classes in "
            f">>{source_root_path}: "
            f"{logs.dict(class_name_count_path)}"
        ),
    )

    # ------------------------------------------------------------------
    # Verify that all requested classes exist
    # ------------------------------------------------------------------
    missing_classes = set(class_names) - set(class_name_count_path.keys())

    if missing_classes:
        raise ValueError(
            f"Missing requested classes: {sorted(missing_classes)}. "
            f"Available classes: {sorted(class_name_count_path.keys())}"
        )

    logs.print("c", "Found all requested class names.")

    # ------------------------------------------------------------------
    # Verify that every requested class contains enough samples
    # ------------------------------------------------------------------
    insufficient_classes = {
        class_name: class_name_count_path[class_name][0]
        for class_name in class_names
        if class_name_count_path[class_name][0]
        < samples_amount_per_all_class
    }

    if insufficient_classes:
        raise ValueError(
            "Not enough samples for the requested classes: "
            f"{insufficient_classes}. "
            f"Required per class: {samples_amount_per_all_class}"
        )

    logs.print(
        "c",
        (
            f"Each requested class contains at least "
            f"{samples_amount_per_all_class} samples."
        ),
    )

    # ------------------------------------------------------------------
    # Create an empty result structure
    # ------------------------------------------------------------------
    class_data_block: _T_pointer2class_dataset = (
        _T__constructor__.creat_empty_input_block_T_label_data(
            labels=class_names
        )
    )

    logs.print("p-s", "Set collection")

    # ------------------------------------------------------------------
    # Select random samples for each requested class
    # ------------------------------------------------------------------
    for label in class_names:
        _, path2_class = class_name_count_path[label]

        # Get files available in the class directory.
        files = [
            file_path
            for file_path in path2_class.iterdir()
            if file_path.is_file()
        ]

        # Select the requested number of unique random files.
        random_files = random.sample(
            files,
            samples_amount_per_all_class,
        )

        name_of_random_files = [
            random_file.name
            for random_file in random_files
        ]

        class_data_block[_T_label(label)]["count"] = (
            samples_amount_per_all_class
        )
        class_data_block[_T_label(label)]["path"] = str(path2_class)
        class_data_block[_T_label(label)]["file_set"] = (
            name_of_random_files
        )

        logs.print("", label)

    logs.print("p-e", "Set collection")
    logs.print("i",f"class_data_block: {logs.dict(class_data_block)}",)

    # ------------------------------------------------------------------
    #  Save the generated dataset description
    # ------------------------------------------------------------------
    logs.storage.save.json(
        filename=Config.DataPackageFileNames.training_images_classes_dataset,
        project_relative_path=save_project_relative_path,
        data=class_data_block,
    )

    return class_data_block
