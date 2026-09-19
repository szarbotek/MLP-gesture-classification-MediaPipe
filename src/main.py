from src.logs import logs

from src.core import *
from src import Config
from src._T_typing import _T_landmarkBatch63

samples_limit_per_class = Config.SAMPLES_LIMIT
save_check_point =  logs.path(f"data/package/{logs.create_name_by_datetime()}")

logs.print('p', "STEP 1 ================================================================")
"""
    Step 1: Random Sampling
    Fetches a balanced set of random sample files for each target class 
    from the raw dataset directory, bounded by the specified per-class limit.
"""

samples = get_random_samples_by_class(
    Config.PROJECT_PATH_TRAINING_DATASET,
    Config.PROJECT_MLP_ACCESSIBLE_CLASSES,
    samples_limit_per_class,
    save_check_point
)

logs.print('p', "STEP 2 ================================================================")
"""
    Step 2: Data Extraction
    Parses collected sample files into structured NumPy arrays containing 
    raw 3D landmark points by MediaPipe Hand landmark and corresponding class labels, saving base checkpoints.
"""

points, labels = get_array_of_class_data(
    samples,
    samples_limit_per_class,
    False,
    save_check_point, ('labels-base', 'points-base', 'class_counter-base'),
)

logs.print('p', "STEP 3 ================================================================")
"""
    Step 3: Landmark Normalization
    Applies spatial transformations (translation, scaling, mirror reflection, 
    and rotation alignment) to standardize the 3D landmark coordinate space.
"""

norm_points: _T_landmarkBatch63|None = None
for flags in [
    #(0,0,0,0), (1,0,0,0), (1,1,0,0), (1,1,1,0),
    (1,1,1,1),
]:
    norm_points: _T_landmarkBatch63 = get_normalization_dataset(
        points, labels,
        save_check_point, "points-norm",
        *flags,
    )
else:
    pass

logs.highlines()
logs.print('p', "STEP 4 ================================================================")
"""
    Step 4: Quality Filtering
    Filters out noisy or low-confidence landmark samples exceeding 
    the defined error threshold to improve overall dataset quality.
    A maximum-distance filter is used for this filtering process.
"""
threshold = -0.25

filtered_points, filtered_labels = get_filtrated_dataset(
    norm_points, labels, threshold,
    save_check_point, ("points-filtrated", "labels-filtrated", "class_counter-filtrated"),
)


logs.highlines()
logs.print('p', "STEP 5 ================================================================")
"""
    Step 5: Model Training
    Initializes and executes the MLP training pipeline using 
    the normalized and filtered dataset, saving model weights and artifacts.
"""
activate_train(filtered_points, filtered_labels, save_check_point)
