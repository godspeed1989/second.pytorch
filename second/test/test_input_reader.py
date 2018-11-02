import fire
import time
import numpy as np
import torch
from second.protos import pipeline_pb2
from google.protobuf import text_format
from torch.utils.data import Dataset
from second.pytorch.builder import box_coder_builder
from second.builder import target_assigner_builder, voxel_builder
from second.builder import dataset_builder
from second.data.preprocess import merge_second_batch

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
    training = True
    dataset = dataset_builder.build(input_cfg, model_cfg,
                                    training, voxel_generator, target_assigner)
    dataset = DatasetWrapper(dataset)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)

    data_iter = iter(dataloader)
    example = next(data_iter)
    print(example.keys())


if __name__ == "__main__":
    fire.Fire()
