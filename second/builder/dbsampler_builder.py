import pickle

import second.core.preprocess as prep
from second.builder import preprocess_builder
from second.core.preprocess import DataBasePreprocessor
from second.core.sample_ops import DataBaseSamplerV2
from second.core.sample_ops_v3 import DataBaseSamplerV3


def build(sampler_config):
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
