import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset')
    parser.add_argument('--seg_level', type=int, default=None)
    parser.add_argument('-s', '--split', default='test')
    parser.add_argument('--fine_grained_annotations', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--subsample_pcd', type=int, default=25_000)
    parser.add_argument('--early_subsample', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--backbone_model', type=str, default='facebook/dinov2-base')
    parser.add_argument('--layer_features', type=int, default=None)
    parser.add_argument('--use_preprocessed_features', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pred_head_path', type=str, default=None)
    parser.add_argument('--store_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--canvas_width', type=int, default=672)
    parser.add_argument('--canvas_height', type=int, default=672)
    parser.add_argument('-fx', type=int, default=1000)
    parser.add_argument('-fy', type=int, default=1000)
    parser.add_argument('-cx', type=float, default=336)
    parser.add_argument('-cy', type=float, default=336)
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-o', '--orientations')
    parser.add_argument('--point_size', type=float, default=3.0)
    parser.add_argument('--light_intensity', type=float, default=0.05)
    parser.add_argument('--interpolation_mode', type=str, default='bicubic')
    parser.add_argument('-k', '--k_means_iterations', type=int, default=30)
    parser.add_argument('--repeat_clustering', type=int, default=5)
    parser.add_argument('--refine_clusters', type=int, default=None)
    parser.add_argument('--samples_per_category', type=int, default=None)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--use_colorized_renders', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--similarity_metric', type=str, default=None)
    parser.add_argument('--k_based_on_class', type=str, default=None)
    parser.add_argument('--feature_reduction', type=str, default=None)
    parser.add_argument('--feature_reduction_dim', type=int, default=None)

    return parser


def parse_args(parser=None):
    if parser is None:
        parser = get_parser()

    return parser.parse_args()
