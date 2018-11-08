import fire
import time
import numpy as np
from google.protobuf import text_format
import torch
from torch.utils.data import Dataset

from second.protos import pipeline_pb2, input_reader_pb2
from second.pytorch.builder import box_coder_builder
from second.builder import target_assigner_builder, voxel_builder

import ipdb
import pickle
import pathlib
from functools import partial, reduce
from second.data.preprocess import merge_second_batch
from second.core import box_np_ops
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.core.point_cloud.bev_ops import points_to_bev
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.dataset import KittiDataset
from second.builder import preprocess_builder
from second.core.preprocess import DataBasePreprocessor

from second.core.sample_ops import DataBaseSamplerV2
from second.core.sample_ops_v3 import DataBaseSamplerV3

from second.utils.check import shape_mergeable
from second.data.preprocess import prep_pointcloud


'''
    from second.builder import dbsampler_builder
'''
def dbsampler_builder_build(sampler_config):
    cfg = sampler_config
    groups = list(cfg.sample_groups)
    prepors = [
        preprocess_builder.build_db_preprocess(c)
        for c in cfg.database_prep_steps
    ]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.database_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler_dict = {
        "DataBaseSamplerV2": DataBaseSamplerV2,
        "DataBaseSamplerV3": DataBaseSamplerV3,
    }
    sample_class = sampler_dict[cfg.database_sampler_name]
    sampler = sample_class(db_infos, groups, db_prepor, rate, grot_range)
    return sampler

'''
    from second.builder import dataset_builder
'''
def dataset_builder_build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          target_assigner=None):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    generate_bev = model_config.generate_bev
    without_reflectivity = model_config.without_reflectivity
    num_point_features = model_config.num_point_features
    out_size_factor = model_config.rpn.layer_strides[0] // model_config.rpn.upsample_strides[0]

    cfg = input_reader_config
    db_sampler_cfg = input_reader_config.database_sampler
    db_sampler = None
    if len(db_sampler_cfg.sample_groups) > 0:  # enable sample
        db_sampler = dbsampler_builder_build(db_sampler_cfg)
    u_db_sampler_cfg = input_reader_config.unlabeled_database_sampler
    u_db_sampler = None
    if len(u_db_sampler_cfg.sample_groups) > 0:  # enable sample
        u_db_sampler = dbsampler_builder_build(u_db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    # [352, 400]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    prep_func = partial(
        prep_pointcloud,
        root_path=cfg.kitti_root_path,
        class_names=list(cfg.class_names),
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels=cfg.max_number_of_voxels,
        remove_outside_points=False,
        remove_unknown=cfg.remove_unknown_examples,
        create_targets=training,
        shuffle_points=cfg.shuffle_points,
        gt_rotation_noise=list(cfg.groundtruth_rotation_uniform_noise),
        gt_loc_noise_std=list(cfg.groundtruth_localization_noise_std),
        global_rotation_noise=list(cfg.global_rotation_uniform_noise),
        global_scaling_noise=list(cfg.global_scaling_uniform_noise),
        global_random_rot_range=list(
            cfg.global_random_rotation_range_per_object),
        db_sampler=db_sampler,
        unlabeled_db_sampler=u_db_sampler,
        generate_bev=generate_bev,
        without_reflectivity=without_reflectivity,
        num_point_features=num_point_features,
        anchor_area_threshold=cfg.anchor_area_threshold,
        gt_points_drop=cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=cfg.remove_points_after_sample,
        remove_environment=cfg.remove_environment,
        use_group_id=cfg.use_group_id,
        out_size_factor=out_size_factor)
    dataset = KittiDataset(
        info_path=cfg.kitti_info_path,
        root_path=cfg.kitti_root_path,
        num_point_features=num_point_features,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func)

    return dataset

class DatasetWrapper(Dataset):
    """ convert our dataset to Dataset class in pytorch.
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        return self._dataset

'''
    test
'''
def test(config_path):
    # cfg
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    input_cfg = config.train_input_reader
    # builds
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    #
    start = time.time()
    input_cfg.database_sampler.database_sampler_name = "DataBaseSamplerV3"
    training = True
    dataset = dataset_builder_build(input_cfg, model_cfg,
                                    training, voxel_generator, target_assigner)
    dataset = DatasetWrapper(dataset)

    print(len(dataset))
    example1 = dataset[2]
    #example2 = dataset[22]
    #example3 = dataset[122]
    print(time.time() - start, 'sec')

if __name__ == "__main__":
    fire.Fire()
