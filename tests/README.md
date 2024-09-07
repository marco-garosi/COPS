# How to benchmark

Run script `19_evaluate_model_various_settings.py` with the following arguments:
```bash
-b=1
--orientations=orientations/004.npz
--dataset=ShapeNetPart
--seg_level=1
--split=test
--backbone_model=facebook/dinov2-base
--layer_features=0
--no-use_preprocessed_features
--no-store_features
--k_means_iterations=50
--repeat_clustering=10
--subsample_pcd=10000
--no-early_subsample
--canvas_width=224
--canvas_height=224
--point_size=0.04
--no-use_colorized_renders
--refine_clusters=7
--cuda
```

> Note: CUDA is required to run the experiments, as datasets can be large (so CPU would be too slow) and some PyTorch3D functionalities require it.


# About orientation files

The point cloud is rendered from multiple viewpoints. The number of views is determined at runtime using a configuration file, which also specifies how to rotate the point cloud in 3D space for each view.

These configuration files are located under the `./orientations` folder (`/tests/orientations`). They are either CSV or NPZ files, with one line per view, where each line determines the orientation of the point cloud in the corresponding view. Angles are expressed in degrees with respect to the x, y, and z axes respectively.
