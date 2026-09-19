"""
    Module responsible for normalizing 3D landmark datasets.
"""

from pathlib import Path
from src.core.landmark_analyzing_tool import normalization_landmarkBatch63
from src._T_typing import (
    _T_landmarkBatch63,
    _T_labelBatch,
    _T_rel_path
)
from src.logs import logs

def get_normalization_dataset(
        points: _T_landmarkBatch63,
        labels: _T_labelBatch,
        save_project_relative_path: _T_rel_path = Path("data/test"),
        save_names: str = "points-norm",

        flag_translation: bool = True,
        flag_scale: bool = True,
        flag_mirror: bool = True,
        flag_rotation: bool = True,
) -> _T_landmarkBatch63:
    """
        Normalizes a batch of 3D landmark data.

        Each sample in the batch is normalized independently using the
        `normalization_landmarkBatch63()` function.

    :param points: Batch of landmark data with shape `(n, 63)`.
        Each sample contains 21 landmarks represented by `(x, y, z)` coordinates.
    :param labels: Batch of corresponding target labels.
    :param save_project_relative_path: Path relative to PROJECT_ROOT where the result is saved.
    :param save_names: Names of saved files.

    :param flag_translation: Enables translation normalization.
    :param flag_scale: Enables scale normalization.
    :param flag_mirror: Enables mirror reflection normalization.
    :param flag_rotation: Enables rotation normalization.
    :return: Normalized batch of landmarks with shape `(n, 63)`.
    """
    logs.endline()
    logs.print( 'p-s',
        f"Normalization started with flags: translation={flag_translation}, "f"scale={flag_scale}, mirror={flag_mirror}, rotation={flag_rotation}"
    )

    timer = logs.stopwatch('normalization_landmarkBatch63')
    timer.run()

    # ------------------------------------------------------------------
    # Perform batch normalization
    # ------------------------------------------------------------------
    norm_data_block: _T_landmarkBatch63 = normalization_landmarkBatch63(
        points, labels,

        flag_translation,
        flag_scale,
        flag_mirror,
        flag_rotation,
    )

    # ------------------------------------------------------------------
    # Save normalized dataset via logs storage
    # ------------------------------------------------------------------
    logs.storage.save.numpy(
        filename=f"{save_names}-{flag_translation}{flag_scale}{flag_mirror}{flag_rotation}",
        project_relative_path=save_project_relative_path,
        data=norm_data_block,
    )

    logs.print('p-e', "Normalization finished successfully.")
    timer.stop()

    return norm_data_block