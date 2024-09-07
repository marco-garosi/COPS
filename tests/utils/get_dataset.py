import config

from data.ScanObjectNN import ScanObjectNN_Part, scanobjectnn_part_collate_fn, Preprocessed_ScanObjectNN_Part, preprocessed_scanobjectnn_part_collate_fn
from data.PartNet import PartNet, Preprocessed_PartNet, preprocessed_partnet_collate_fn
from data.PartNetSemanticSegmentation import PartNetSemanticSegmentation, preprocessed_partnet_sem_seg_collate_fn
from data.PartNetMobility import PartNetMobility_Part, Preprocessed_PartNetMobility_Part, preprocessed_partnet_mobility_collate_fn
from data.PartNetE import PartNetE, partnete_collate_fn
from data.ShapeNetPart import ShapeNetPart
from data.FAUST import FAUST


def get_dataset(args):
    collate_fn, dataset, preprocessed_folder = None, None, None

    # No dataset specified in args
    if args.dataset is None:
        print('--dataset not specified')

    # ScanObjectNN Part
    elif args.dataset == 'ScanObjectNN-Part':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_SCANOBJECTNN_PART
        if not args.use_preprocessed_features:
            dataset = ScanObjectNN_Part(config.PATH_TO_SCANOBJECTNN_PART, split=args.split, filter_background=False)
            collate_fn = scanobjectnn_part_collate_fn
        else:
            dataset = Preprocessed_ScanObjectNN_Part(preprocessed_folder, config.PATH_TO_SCANOBJECTNN_PART, args.backbone_model.split('/')[-1], split=args.split)
            collate_fn = preprocessed_scanobjectnn_part_collate_fn

    # PartNet
    elif args.dataset == 'PartNet':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_PARTNET
        if not args.use_preprocessed_features:
            dataset = PartNet(config.PATH_TO_PARTNET, split=args.split)
        else:
            dataset = Preprocessed_PartNet(preprocessed_folder, config.PATH_TO_PARTNET,
                                           args.backbone_model.split('/')[-1], split=args.split)
            collate_fn = preprocessed_partnet_collate_fn

    # PartNet for Semantic Segmentation
    elif args.dataset == 'PartNet-SemSeg':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_PARTNET_SEMSEG
        if not args.use_preprocessed_features:
            dataset = PartNetSemanticSegmentation(config.PATH_TO_PARTNET_SEMANTIC_SEGMENTATION, config.PATH_TO_PARTNET,
                                                  level=str(args.seg_level), split=args.split)
        else:
            pass

    # PartNet Mobility
    elif args.dataset == 'PartNetMobility':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_PARTNET_MOBILITY
        if not args.use_preprocessed_features:
            dataset = PartNetMobility_Part(config.PATH_TO_PARTNET_MOBILITY, split=args.split)
        else:
            dataset = Preprocessed_PartNetMobility_Part(preprocessed_folder, config.PATH_TO_PARTNET_MOBILITY,
                                                        args.backbone_model.split('/')[-1], split=args.split)
            collate_fn = preprocessed_partnet_mobility_collate_fn

    # PartNet Mobility for Semantic Segmentation
    elif args.dataset == 'PartNetE-SemSeg':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_PARTNETE
        if not args.use_preprocessed_features:
            dataset = PartNetE(config.PATH_TO_PARTNETE, semantic_segmentation=True, split=args.split)
            collate_fn = partnete_collate_fn
        else:
            pass

    # ShapeNet Part
    elif args.dataset == 'ShapeNetPart':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_SHAPENET_PART
        if not args.use_preprocessed_features:
            dataset = ShapeNetPart(config.PATH_TO_SHAPENET_PART, split=args.split)
        else:
            print('Not supported yet')
            exit()

    # FAUST
    elif args.dataset == 'FAUST':
        preprocessed_folder = config.PATH_TO_PREPROCESSED_FAUST
        if not args.use_preprocessed_features:
            dataset = FAUST(config.PATH_TO_FAUST, fine_grained_annotations=args.fine_grained_annotations, split=args.split)
        else:
            print('Not supported yet')
            exit()

    # Invalid dataset
    else:
        print('--dataset not valid')

    return dataset, collate_fn, preprocessed_folder
