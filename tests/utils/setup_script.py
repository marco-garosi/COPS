import os.path

import torch
import numpy as np
import json
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from source.models.Predictor import FTCC

from .parse_arguments import parse_args
from .get_dataset import get_dataset

import config


def setup(
        is_pred_head_path_required=False,
        is_orientations_required=False
):
    args = parse_args()

    if is_pred_head_path_required:
        assert args.pred_head_path is not None, '--pred_head_path is required'

    dataset, collate_fn, preprocessed_folder = get_dataset(args)
    if dataset is None:
        exit()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=config.NUM_WORKERS_DATALOADER, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    # Load the backbone model
    model = None
    image_processor = None
    if not args.use_preprocessed_features:
        print('Loading backbone model')
        image_processor = AutoProcessor.from_pretrained(args.backbone_model)
        model = AutoModel.from_pretrained(args.backbone_model, output_hidden_states=args.layer_features is not None)
        model.eval()
        model.to(device)
        print(f'Backbone model loaded to: {device}')

    # Embedding dimension of the backbone
    embedding_dimension = 768
    if args.backbone_model == 'facebook/dinov2-base':
        embedding_dimension = 768

    if is_orientations_required and args.orientations is None:
        print('--orientations not specified')
        exit()
    orientations = None
    if args.orientations is not None:
        extension = os.path.splitext(args.orientations)[1]
        if extension == '.csv':
            orientations = np.genfromtxt(args.orientations, delimiter=',')
        elif extension == '.npz':
            orientations = np.load(args.orientations)

    prediction_head = None
    if args.pred_head_path is not None:
        # Instantiate the head
        prediction_head = FTCC(
            input_dim=embedding_dimension * len(orientations),
            output_dim_class=len(dataset.class_ids),
            output_dim_k=len(dataset.part_ids),
            hidden_dim=embedding_dimension
        )
        prediction_head.load_state_dict(torch.load(args.pred_head_path))
        prediction_head.to(device)
        prediction_head.eval()

    return args, model, image_processor, embedding_dimension, prediction_head, orientations, dataset, dataloader, preprocessed_folder, device
