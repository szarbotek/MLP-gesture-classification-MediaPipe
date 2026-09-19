import numpy as np

from src.logs import logs
from src._T_typing import (
    _T_landmarkBatch63,
    _T_labelBatch,
    _T_classBatch,
    _T_rel_path,
    _T_label,
)


from src.core.landmark_analyzing_tool import fragmentation_landmarkBatch63_by_label, filter_by_minimum_distance

def get_filtrated_dataset(
        points: _T_landmarkBatch63,
        labels: _T_labelBatch,
        threshold: float|int,

        save_project_relative_path: _T_rel_path = logs.path("data/test"),
        save_names: tuple[str, str, str] = ("points-filtrated", "labels-filtrated", "class_counter-filtrated"),

) -> tuple[_T_landmarkBatch63, _T_labelBatch]:
    """
        Filters a landmark dataset based on the minimum distance between samples.

        The input dataset is divided into separate classes based on their labels.
        Each class is filtered independently using the specified minimum-distance
        threshold. The filtered class batches are then combined into a single
        dataset. The filtered data and class statistics are also saved to disk.

    :param points: Batch of 3D landmark data with shape `(n, 63)`.
        Each sample contains 21 landmarks represented by `(x, y, z)`
        coordinates.
    :param labels: Labels corresponding to the samples in `points`.
    :param threshold: Minimum distance threshold used to filter the samples.

    :param save_project_relative_path: Project-relative path where the
        filtered dataset and class statistics are saved.
    :param save_names: Names of saved files.
    :return: A tuple containing the filtered landmark batch and the
        corresponding label batch.
    """

    logs.print('p', 'filtration process')
    logs.print('p-s', 'generating dataset')

    # Start the global filtration timer and initialize the processing report.
    timer = logs.stopwatch("filtration")
    timer.run()
    report = logs.report( {"label": str, "count": int, "time": float} )

    # Divide the input dataset into separate batches based on their labels.
    class_data_block: _T_classBatch = fragmentation_landmarkBatch63_by_label(
        points=points, labels=labels
    )
    logs.print('i', 'separate class batch')

    # ------------------------------------------------------------------
    # Store the filtered samples and the number of samples remaining in each class.
    # ------------------------------------------------------------------

    class_dataset_after_filtration: _T_classBatch = {}
    class_labels_counter: dict[_T_label, int] = {lb: 0 for lb in class_data_block.keys()}

    for lb, pt in class_data_block.items():
        # Measure the processing time for the current class.
        special_timer = logs.stopwatch(lb)
        special_timer.run()

        # Filter the current class using the specified minimum-distance threshold.
        grupe_filtered, thval = filter_by_minimum_distance(pt, threshold)

        class_dataset_after_filtration[lb] = grupe_filtered
        logs.print('s', f"processed label {lb}")
        special_timer.stop()

        # Add the filtration results for the current class to the processing report.
        report.put( "label", lb )
        report.put( "count", len(grupe_filtered) )
        report.put( "time", special_timer.elapsed )

        class_labels_counter[lb] = len(grupe_filtered)


    logs.print('p-e', 'generating dataset')
    timer.stop()

    # Display the processing report after all classes have been filtered.
    report.show()
    report.saveSelf("filtrated_dataset",save_project_relative_path, )

    # Combine all filtered class batches into a single points array and
    # generate the corresponding label array.
    ret_points = np.concatenate(list(class_dataset_after_filtration.values()), axis=0)
    ret_labels = np.asarray([lb for lb, pt in class_dataset_after_filtration.items() for _ in range(len(pt))])

    # ------------------------------------------------------------------
    # Save the filtered points, labels, and class sample counts.
    # ------------------------------------------------------------------
    logs.storage.save.numpy(
        save_names[0],
        save_project_relative_path,
        ret_points
    )
    logs.storage.save.numpy(
        save_names[1],
        save_project_relative_path,
        ret_labels
    )
    logs.storage.save.json(
        save_names[2],
        save_project_relative_path,
        class_labels_counter,
    )

    return ret_points, ret_labels