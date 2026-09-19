import numpy as np

from src._T_typing import (
    _T_landmark63,
    _T_landmarkBatch63,
    _T_landmark21xyz,
    _T_landmarkBatch21xyz,

    _T_labelBatch,
    _T_label,
    _T_classBatch,
    _T_xyz,
    _T_xyzBatch,

    _T_vector0xyz,
    _T_vectorBatch0xyz,

    _T_angle_rad,
    _T_angle_degree,
    _T_label,
    _T__converter__,
)

from mediapipe.tasks.python.components.containers import NormalizedLandmark
from sklearn.cluster import DBSCAN


def convert_NormalizedLandmark_2_array(landmark: list[NormalizedLandmark]) -> _T_landmark63:
    """
    Converts a list of MediaPipe landmarks into a flattened NumPy array.

    Each landmark is represented by three coordinates `(x, y, z)`.
    The resulting array contains 21 landmarks flattened into a single
    vector of 63 values.

    :param landmark: List of 3D MediaPipe landmarks.
    :return: Flattened landmark vector with shape `(63,)`.
    """
    points: _T_landmark21xyz = np.array(
        [
            [lm.x, lm.y, lm.z] for lm in landmark
            # [x1, y1, z1]
            # [x2, y2, z2]
            # ...
            # [x21, y21, z21]
        ],
        dtype=np.float64
    )
    # Flatten the landmark array into a single vector.
    return points.reshape(-1)


class CoordinateTransform:
    mirrorX = np.array([-1,  1,  1])
    mirrorY = np.array([ 1, -1,  1])
    mirrorZ = np.array([ 1,  1, -1])

    axisX = np.array([1,  0,  0])
    axisY = np.array([0,  1,  0])
    axisZ = np.array([0,  0,  1])

    originX = np.array([1,  0,  0])
    originY = np.array([0,  1,  0])
    originZ = np.array([0,  0,  1])

    originRevX = np.array([-1, 0, 0])
    originRevY = np.array([0, -1, 0])
    originRevZ = np.array([0, 0, -1])

    rotXY = lambda angle_rad: np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    rotXZ = lambda angle_rad: np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

    rotYZ = lambda angle_rad: np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])


## =====================================================================================================================
## normalization
## =====================================================================================================================

def normalization_landmark63(
    points: _T_landmark63,
    label: _T_label,
    flag_translation: bool = True,
    flag_scale: bool = True,
    flag_mirror: bool = True,
    flag_rotation: bool = True,
) -> _T_landmark63:
    """
    Normalizes a set of 21 3D landmarks represented by 63 values.

    Each landmark is represented by three coordinates `(x, y, z)`.

    The normalization process consists of four consecutive steps:

    1. Translation relative to the Wrist landmark.
       All landmarks are translated so that the Wrist landmark (0) is
       positioned at the origin of the coordinate system `(0, 0, 0)`.
       This removes the influence of the hand's global position.

    2. Hand orientation normalization.
       The landmark coordinates are transformed so that the representation
       is independent of whether the detected hand is the left or right hand.
       For the left hand, an appropriate reflection is applied to match the
       reference representation of the right hand.

    3. Rotation normalization based on the Wrist -> MiddleFingerMCP segment.
       The vector between the Wrist landmark (0) and MiddleFingerMCP landmark
       (9) is used to determine the hand orientation. The landmarks are then
       rotated in the XY plane to match the selected reference orientation.

    4. Scale normalization based on the Wrist -> MiddleFingerMCP segment.
       The distance between landmarks 0 and 9 is used as the reference length.
       All landmarks are scaled relative to this distance, making the resulting
       representation independent of hand size and distance from the camera.

    :param points: 21 landmarks represented as a flattened vector with shape
        `(63,)`.
    :param label: Label assigned to the input landmark sample.
    :param flag_translation: Enables translation normalization.
    :param flag_scale: Enables scale normalization.
    :param flag_mirror: Enables mirror normalization.
    :param flag_rotation: Enables rotation normalization.
    :return: Normalized landmark data represented as a flattened vector with
        shape `(63,)`.
    """

    points: _T_landmark21xyz = _T__converter__.conv_landmark63_to_landmark21xyz(points)

    # Mirror the point cloud along the Y axis.
    points: _T_landmark21xyz = points * CoordinateTransform.mirrorY

    # Normalize the landmarks relative to the Wrist origin point.
    if flag_translation:
        pt_wrist: _T_xyz = points[0,:]
        points: _T_landmark21xyz  = points - pt_wrist

    # Scale the landmarks relative to the Wrist -> MiddleFingerMCP segment.
    if flag_scale:
        lenght_Wrist_2_MiddleFingerMCP: _T_vector0xyz = points[0,:] - points[9,:]
        scale_factor = 1 / np.linalg.norm( lenght_Wrist_2_MiddleFingerMCP )
        points: _T_landmark21xyz = points * scale_factor

    # Transform the landmarks into the representation of the right hand.
    # The direction of the vector between the LittleFinger and RingFinger
    # fingertips is used as the reference.
    if flag_mirror:
        pt_wrist: _T_xyz = points[0, :]
        pt_direction: _T_xyz = (points[5, :] + points[17, :]) / 2

        direction_hand: _T_vector0xyz = pt_direction - pt_wrist
        vector_originY: _T_vector0xyz = CoordinateTransform.originY

        # Calculate the signed angle between the hand direction vector
        # and the reference vector.
        cos_theta = np.dot(direction_hand, vector_originY) / (np.linalg.norm(direction_hand) * np.linalg.norm(vector_originY))
        sin_theta = np.dot( np.cross(vector_originY, direction_hand), CoordinateTransform.axisZ)

        # Calculate the signed angle in the range from -pi to +pi.
        theta: _T_angle_rad = np.arctan2(sin_theta, cos_theta)

        if (-np.pi / 4 <= theta <= np.pi / 4):
            # Determine whether the hand needs to be mirrored along the X axis.
            pt_IndexFingerTIP: _T_vector0xyz  = points[8, :]
            pt_PinkyTIP: _T_vector0xyz  = points[20, :]
            direction_pointer: _T_vector0xyz = pt_PinkyTIP - pt_IndexFingerTIP

            # Check the direction along the X axis.
            if direction_pointer[0] < 0:
                points = points * CoordinateTransform.mirrorX
            else: pass
        elif (np.pi / 4 < theta <= 3 * np.pi / 4):
            # Mirror the landmarks along the X axis.
            points = points * CoordinateTransform.mirrorX
        elif (-3 * np.pi / 4 <= theta < -np.pi / 4):
            # The current orientation does not require mirroring.
            pass
        else:
            # No action is taken for this orientation.
            # Such an orientation is probably outside the expected gesture set.
            pass

    if flag_rotation:
        pt_Wrist: _T_xyz = points[0, :]
        pt_direction: _T_xyz = (points[5, :] + points[17, :]) / 2
        direction_hand: _T_vector0xyz = pt_direction - pt_Wrist
        vector_originY: _T_vector0xyz = CoordinateTransform.originY

        cos_theta = np.dot(direction_hand, vector_originY) / (
                    np.linalg.norm(direction_hand) * np.linalg.norm(vector_originY))
        sin_theta = np.dot(np.cross(vector_originY, direction_hand), CoordinateTransform.axisZ)
        theta: _T_angle_rad = np.arctan2(sin_theta, cos_theta)

        def _align_to_reference(vector_ref: _T_vector0xyz) -> _T_landmark21xyz:
            """
            Rotates the landmarks to align the hand with a reference vector.

            The angle between the Wrist -> MiddleFingerMCP vector and the
            reference vector is calculated. The landmarks are then rotated
            by this angle in the XY plane.

            :param vector_ref: Reference vector used to determine the target
                orientation.
            :return: Rotated landmark data with shape `(21, 3)`.
            """
            pt_Wrist: _T_xyz = points[0, :]
            pt_MiddleFingerMCP: _T_xyz = points[9, :]
            vector_Wrist_2_MiddleFingerMCP: _T_vector0xyz = pt_MiddleFingerMCP - pt_Wrist
            cos_theta: np.float64 = np.dot(vector_Wrist_2_MiddleFingerMCP, vector_ref) / (
                    np.linalg.norm(vector_Wrist_2_MiddleFingerMCP) * np.linalg.norm(vector_ref))
            stand_cos_theta: np.float64 = np.clip(cos_theta, -1.0, 1.0)
            theta: _T_angle_rad = np.arccos(stand_cos_theta)
            rotXY = CoordinateTransform.rotXY(theta)
            return points @ rotXY.T

        """
            Determine the section in which the hand direction vector is located.
        """
        if (-np.pi / 4 <= theta <= np.pi / 4):
            # UP
            # Case for gestures directed upwards, such as one, peace, and fist.
            points: _T_landmark21xyz = _align_to_reference(CoordinateTransform.originY)
        elif (-3 * np.pi / 4 <= theta < -np.pi / 4):
            # RIGHT
            # Case for gestures shown from the side, such as like and unlike.
            points: _T_landmark21xyz = _align_to_reference(CoordinateTransform.originX)
        elif (np.pi / 4 < theta <= 3 * np.pi / 4):
            # LEFT
            pass
            # The left orientation should not occur due to normalization
            # to the right-hand representation.
        elif (3 * np.pi / 4 < theta <= np.pi) or (-np.pi <= theta < -3 * np.pi / 4):
            # DOWN
            pass
            # Downward-facing gestures are not present due to the characteristics
            # of the dataset.
        else:
            raise Exception("To big theta")

    points: _T_landmark63 = _T__converter__.conv_landmark21xyz_to_landmark63(points)

    return points


def normalization_landmarkBatch63(
    data_block: _T_landmarkBatch63,
    label_block: _T_labelBatch,

    flag_translation: bool = True,
    flag_scale: bool = True,
    flag_mirror: bool = True,
    flag_rotation: bool = True,
) -> _T_landmarkBatch63:
    """
    Normalizes a batch of 3D landmark data.

    Each sample in the batch is normalized independently using the
    `normalization_landmark63()` function.

    :param data_block: Batch of landmark data with shape `(n, 63)`.
        Each sample contains 21 landmarks represented by `(x, y, z)`
        coordinates.
    :param label_block: Labels corresponding to the samples in `data_block`.
    :param flag_translation: Enables translation normalization.
    :param flag_scale: Enables scale normalization.
    :param flag_mirror: Enables mirror normalization.
    :param flag_rotation: Enables rotation normalization.
    :return: Normalized landmark batch with shape `(n, 63)`.
    """

    # logs.endline()
    # logs.print('p-s', "Normalization with flags {}{}{}{}".format( flag_translation, flag_scale, flag_mirror, flag_rotation))
    # timer = logs.stopwatch('normalization_landmarkBatch63')
    # timer.run()

    norm_data_block: _T_landmarkBatch63 = np.array(
        [
            normalization_landmark63(points, label, flag_translation, flag_scale, flag_mirror, flag_rotation)
                for points, label in zip(data_block, label_block)
        ],
        dtype=np.float64
    )

    # logs.save.numpy(
    #     filename="norm-points-{}{}{}{}".format(flag_translation,flag_scale,flag_mirror,flag_rotation),
    #     project_path=logs.path("data/MLP/array/norm"),
    #     data=norm_data_block
    # )
    # logs.print('p-e', "Finish normalization.")
    # timer.stop()

    return norm_data_block


def fragmentation_landmarkBatch63_by_label(
    points: _T_landmarkBatch63,
    labels: _T_labelBatch,
) -> _T_classBatch:
    """
    Splits a landmark batch into separate groups based on their labels.

    Each unique label is used as a key in the returned dictionary, with the
    corresponding landmark samples stored as a NumPy array.

    :param points: Batch of landmark data with shape `(n, 63)`.
    :param labels: Labels corresponding to the samples in `points`.
    :return: Dictionary containing a separate landmark batch for each label.
    """

    class_block: dict[_T_label, list[_T_landmark63]] = {}

    for lb, pt in zip(labels, points):
        if lb not in class_block:
            class_block[lb] = []

        class_block[lb].append(pt)

    return {
        lb: np.asarray(pts, dtype=np.float64)
        for lb, pts in class_block.items()
    }


## =====================================================================================================================
## additional features for geometric centers
## =====================================================================================================================

def geometricalCenter_landmark63(landmark: _T_landmark63) -> _T_xyz:
    """
    Calculates the geometric center of a single landmark set.

    The geometric center is calculated as the mean of all 21 landmark
    coordinates.

    :param landmark: Landmark data containing 21 3D points in flattened form.
    :return: Geometric center represented by an `(x, y, z)` coordinate.
    """
    points: _T_landmark21xyz = _T__converter__.conv_landmark63_to_landmark21xyz(landmark)
    gc: _T_xyz = np.mean(
        points,
        axis=0
    )
    return gc


def geometricalCenterBatch_landmarkBatch63(points: _T_landmarkBatch63) -> _T_xyzBatch:
    """
    Calculates the geometric center for each landmark set in a batch.

    :param points: Batch of landmark data with shape `(n, 63)`.
    :return: Batch of geometric centers with shape `(n, 3)`.
    """
    points: _T_landmarkBatch21xyz = _T__converter__.conv_landmarkBatch63_to_landmarkBatch21xyz(points)
    points_mean: _T_xyzBatch = np.mean(points, axis=1)

    return points_mean


def geometricalCenterBatch_characteristicPoints(points: _T_landmarkBatch63) -> _T_landmark63:
    """
    Calculates the resulting landmark representation for the entire population.

    The resulting landmark is calculated as the mean of all corresponding
    landmark coordinates across the entire batch.

    :param points: Batch of landmark data with shape `(n, 63)`.
    :return: Mean landmark representation with shape `(63,)`.
    """

    landmark: _T_landmark63 = np.mean(points, axis=0)
    return landmark


def geometricalCenter_landmarkBatch63(points: _T_landmarkBatch63) -> _T_xyz:
    """
    Calculates the geometric center of the entire landmark population.

    The geometric center is calculated from the geometric centers of all
    individual samples.

    :param points: Batch of landmark data with shape `(n, 63)`.
    :return: Geometric center of the entire population as an `(x, y, z)`
        coordinate.
    """

    chr_points: _T_xyzBatch = geometricalCenterBatch_landmarkBatch63(points)
    gc: _T_xyz = np.mean(
        chr_points,
        axis=0
    )
    return gc


def centroid(data: _T_xyzBatch, eps =0.5, min_samples = 10 ) -> _T_xyz:
    """
    Finds the geometric center of the largest cluster of points.

    DBSCAN is used to identify clusters in the input data. Noise points are
    excluded, and the geometric center of the largest remaining cluster is
    returned.

    :param data: Batch of 3D points used for clustering.
    :param eps: Maximum distance between samples for them to be considered
        part of the same neighborhood.
    :param min_samples: Minimum number of samples required to form a cluster.
    :return: Geometric center of the largest detected cluster.
    """

    mask = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(data)

    # Select the point groups corresponding to the detected cluster labels.
    clusters = [
        data[mask == m]
            for m in set(mask)
        if m != -1
    ]

    # Select the cluster containing the largest number of points.
    main_cluster = max(clusters, key=len)

    # Calculate the geometric center of the largest cluster.
    center = np.mean(main_cluster, axis=0)

    return center


def centroid_geometricalCenterBatch_landmarkBatch63(points: _T_landmarkBatch63, eps =0.5, min_samples = 50) -> _T_xyz:
    """
    Calculates the center of the largest cluster of geometric centers.

    Depending on the `min_samples` value, the function either calculates the
    geometric centers of all samples and applies clustering or directly
    calculates the geometric center of the entire landmark population.

    :param points: Batch of landmark data with shape `(n, 63)`.
    :param eps: Maximum distance between samples for them to be considered
        part of the same neighborhood.
    :param min_samples: Minimum number of samples required to form a cluster.
    :return: Center of the largest cluster as an `(x, y, z)` coordinate.
    """
    if min_samples >= 50:
        gc_points = geometricalCenterBatch_landmarkBatch63(points)
        center: _T_xyz = centroid(gc_points, eps, min_samples)
    else:
        center: _T_xyz = geometricalCenter_landmarkBatch63(points)

    return center


## =====================================================================================================================
## filtration
## =====================================================================================================================

def filter_by_minimum_distance(
        points: _T_landmarkBatch63,
        threshold: float = 0.25,
) -> tuple[_T_landmarkBatch63, float]:
    """
        Filters landmark samples based on their distance from the population center.

        The geometric center of each landmark sample is calculated first. The
        population center is then determined using clustering. Samples whose
        standardized distance from the population center exceeds the specified
        threshold are removed.

    :param points: Batch of landmark data with shape `(n, 63)`.
    :param threshold: Threshold used to determine which samples are retained.
    :return: A tuple containing the filtered landmark batch and the calculated
        filtration distance range.
    """

    # poins: _T_landmarkBatch21xyz = _T__converter__.conv_landmarkBatch63_to_landmarkBatch21xyz(points)

    gc_of_points:  _T_xyzBatch = geometricalCenterBatch_landmarkBatch63(points)
    gc_population: _T_xyz = centroid_geometricalCenterBatch_landmarkBatch63(points, eps=0.5, min_samples=int(len(points)*0.5) )

    distances: _T_vectorBatch0xyz = np.linalg.norm(gc_of_points - gc_population, axis=1)
    std_distance = (distances - np.mean(distances)) / np.std(distances, ddof=1)
    filtration_range = np.mean(distances) + threshold * np.std(distances, ddof=1)

    # Select the samples that satisfy the filtration threshold.
    mask =  std_distance <= threshold
    accept_points: _T_landmarkBatch63 = points[mask]

    return accept_points, filtration_range

## =====================================================================================================================
## train
## =====================================================================================================================
