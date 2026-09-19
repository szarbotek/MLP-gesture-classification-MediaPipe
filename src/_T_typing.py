from typing import NewType, TypedDict, TypeAlias

import numpy as np
from numpy.typing import NDArray

_T_label = NewType("_T_label_name", str)

class _T_pointerValues(TypedDict):
    count: int
    path: str
    file_set: list[str]


_T_pointer2class_dataset = dict[_T_label, _T_pointerValues]

from typing import Annotated

_T_landmark63          = Annotated[ NDArray[np.float64], "(63,)", ]
_T_landmarkBatch63     = Annotated[ NDArray[np.float64], "(n, 63)",]
_T_landmark21xyz       = Annotated[ NDArray[np.float64], "(21, 3)", ]
_T_landmarkBatch21xyz  = Annotated[ NDArray[np.float64], "(n, 21, 3)",]

_T_xyz                 = Annotated[ NDArray[np.float64], "(3,)", ]
_T_xyzBatch            = Annotated[ NDArray[np.float64], "(n, 3)", ]

_T_vector0xyz          = Annotated[ NDArray[np.float64], "(3,)", ]
_T_vectorBatch0xyz     = Annotated[ NDArray[np.float64], "(n, 3)", ]

_T_angle_degree        = np.float64
_T_angle_rad           = np.float64

_T_labelBatch        = Annotated[ NDArray[np.str_], "(n, 1)",]

_T_classBatch        = dict[_T_label, _T_landmarkBatch63]

from pathlib import Path

_T_path             = Path # ścieżka do miejsca
_T_target_path      = Path # ścieżka do pliku

_T_root             = Path # punkt poczatku relatywności
_T_rel_path         = Path # ścieżka relatywna do miejsca
_T_rel_target_path  = Path # ścieżka relatywna do pliku


## === PLOT ===

from collections.abc import Mapping

_T_colorHex =  Annotated[ str, "ABCDEF",]
_T_cmap_colorHex = Mapping[_T_label, _T_colorHex]


## =====

class _T__converter__:
    @staticmethod
    def conv_landmark63_to_landmark21xyz( data: _T_landmark63 ) -> _T_landmark21xyz:
        # return data.reshape( -1, 21, 3 )
        return data.reshape( -1, 3 )

    @staticmethod
    def conv_landmark21xyz_to_landmark63( data: _T_landmark21xyz ) -> _T_landmark63:
        return data.reshape( -1 )

    @staticmethod
    def conv_landmarkBatch63_to_landmarkBatch21xyz( data: _T_landmarkBatch63 ) -> _T_landmarkBatch21xyz:
        return data.reshape( -1, 21, 3 )

    @staticmethod
    def conv_landmarkBatch21xyz_to_landmarkBatch63( data: _T_landmarkBatch21xyz ) -> _T_landmarkBatch63:
        return data.reshape( -1, 63 )

    @staticmethod
    def conv_landmarkAny_to_xyzBatch( data: NDArray[np.float64] ) -> _T_xyz:
        return data.reshape( -1, 3 )

class _T__constructor__:

    @staticmethod
    def create_empty_input_block_T_label_values() -> _T_pointerValues:
        return {
            "count": 0,
            "path": "",
            "file_set": [],
        }

    @staticmethod
    def creat_empty_input_block_T_label_data(labels: list[str | _T_label]) -> _T_pointer2class_dataset:
        return {
            _T_label(l): _T__constructor__.create_empty_input_block_T_label_values()
            for l in labels
        }




