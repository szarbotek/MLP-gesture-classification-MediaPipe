"""
    The program transforms selected image samples into landmark data.

    Landmarks are extracted directly from the source images without
    applying any normalization during the landmark detection stage.
"""
from pathlib import Path
from idlelib.colorizer import prog_group_name_to_tag
from typing import Any

from numpy import dtype, float64, ndarray, str_

from src.logs import logs
from src.Config import PROJECT_PATH_MEDIAPIPE_MODEL, set_up

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision
from src.core.landmark_analyzing_tool import convert_NormalizedLandmark_2_array

from src._T_typing import (
    _T_label,
    _T_pointer2class_dataset,
    _T_landmark63,
    _T_landmarkBatch63,
    _T_labelBatch,
    _T_rel_path
)
from numpy.typing import NDArray


def get_array_of_class_data(
    class_data_block: _T_pointer2class_dataset,
    limit_per_class: int,
    flag_break_on_limit: bool = True,

    save_project_relative_path: _T_rel_path = Path("data/test"),
    save_filenames: tuple[str, str, str] = ('labels-base', 'points-base', 'class_counter-base'),

) -> tuple[_T_labelBatch, _T_landmarkBatch63]:
    """
       The function uses the MediaPipe Hand Landmarker model
       (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=pl)
       to detect hand landmarks in the selected image samples.

       For every processed image, the detected hand landmarks are converted
       into numerical arrays containing 63 values per detected hand.

       The function processes the images grouped by class labels and stores
       the resulting landmark data together with their corresponding labels.

       The operation returns two NumPy arrays saved as checkpoints:
       one containing the class labels and one containing the corresponding
       63-dimensional landmark vectors.

       :param class_data_block: Input dataset description containing class
           labels, source paths and selected file names.
       :param limit_per_class: Maximum number of detected hands stored
           for each class.
       :param flag_break_on_limit: Determines whether processing of a class
           should stop after reaching the specified limit.
       :param save_project_relative_path: Path to the directory where the
           resulting checkpoints should be saved.
       :param save_filenames: Names of the output files used for storing
           the landmark points and class labels.
       :param save_nametag: Optional suffix added to the names of the
           generated output files.
       :return: (labels, points) NumPy arrays containing the class labels
           and their corresponding 63-dimensional landmark vectors.
    """

    logs.print('d', "get_array_of_class_data")
    logs.print('p-s', "Generating table of landmark points")

    stopwatch = logs.stopwatch("get_array_of_class_data")
    stopwatch.run()
    report = logs.report({"label": str, "count": int, "time": float})

    ## list of consecutive labels
    labels: NDArray | None = None

    ## list of consecutive sets of 63-dimensional vectors
    points: NDArray | None = None

    class_labels_counter: dict[_T_label, int] = {
        lb: 0 for lb in class_data_block.keys()
    }

    ## setup environment variables required for GPU acceleration
    set_up.environ()

    ## configure GPU acceleration and load the model
    base_options = BaseOptions(
        model_asset_path=str(PROJECT_PATH_MEDIAPIPE_MODEL),
        delegate=BaseOptions.Delegate.GPU
    )

    ## configure the main detection parameters:
    ## detection of up to two hands and image processing mode
    main_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_presence_confidence=0.35,
        min_hand_detection_confidence=0.35,
        min_tracking_confidence=0.5,
    )

    result_labels: list[_T_label] = []
    result_points: list[_T_landmark63] = []

    # ------------------------------------------------------------------
    # Initialize the model using the configured options
    # ------------------------------------------------------------------
    try:
        with vision.HandLandmarker.create_from_options(
            main_options
        ) as landmark:

            ## process all class labels
            for label, data in class_data_block.items():
                count, path, file_set = data.values()

                logs.print('s', f"Process label: {label}")
                unit_stopwatch = logs.stopwatch(label)
                unit_stopwatch.run()

                ## process images belonging to the current class
                for f_name in file_set:
                    source_path_2_image = Path(path) / Path(f_name)

                    result = None

                    ## try to process the image; continue if an error occurs
                    try:
                        image = mp.Image.create_from_file(
                            str(source_path_2_image)
                        )
                        result = landmark.detect(image)

                    except Exception as e:
                        logs.print(
                            'e',
                            f"Problem with file {f_name}; {e}"
                        )
                        continue

                    ## check whether any hands were detected
                    check_results = len(result.handedness)

                    ## no hands detected - skip the current image
                    if check_results == 0:
                        continue

                    ## process the landmarks detected for each hand
                    for normalized_landmark in result.hand_landmarks:
                        numeric_points: _T_landmark63 = (
                            convert_NormalizedLandmark_2_array(
                                normalized_landmark
                            )
                        )

                        result_labels.append(label)
                        result_points.append(numeric_points)
                        class_labels_counter[label] += 1

                        ## stop processing the current class after
                        ## reaching the requested limit
                        if (
                            flag_break_on_limit
                            and class_labels_counter[label]
                            >= limit_per_class
                        ):
                            break

                    ## break the image loop when the class limit is reached
                    if (
                        flag_break_on_limit
                        and class_labels_counter[label]
                        >= limit_per_class
                    ):
                        break

                logs.print(
                    'i',
                    (
                        f"End processing: {label}, "
                        f"recognized {class_labels_counter[label]} images"
                    )
                )

                unit_stopwatch.stop()
                logs.endline()

                ## add processing information to the report
                report.put("label", label)
                report.put("count", class_labels_counter[label])
                report.put("time", unit_stopwatch.elapsed)

                del unit_stopwatch

    except Exception as e:
        logs.print('e', f"Model cannot be initialized: {e}")

    logs.print('i', "Model processed successfully")

    labels: _T_labelBatch = np.asarray(
        result_labels,
        dtype=np.str_
    )

    points: _T_landmarkBatch63 = np.asarray(
        result_points,
        dtype=np.float64
    )

    logs.print('i', f"Create data block: {points.shape}")

    stopwatch.stop()

    report.show()
    report.saveSelf( "array_of_class_data", save_project_relative_path )

    name_labels, name_points, name_class_counter = save_filenames

    logs.endline()

    # ------------------------------------------------------------------
    #  Save the generated dataset description
    # ------------------------------------------------------------------
    logs.storage.save.numpy(
        name_labels,
        save_project_relative_path,
        labels,
    )

    logs.storage.save.numpy(
        name_points,
        save_project_relative_path,
        points,
    )

    logs.storage.save.json(
        name_class_counter,
        save_project_relative_path,
        class_labels_counter,
    )

    logs.print( 'p-e',"Generating table of landmark data described by class labels" )

    return points, labels
