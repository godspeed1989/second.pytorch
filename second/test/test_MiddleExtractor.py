import fire
import numpy as np
import torch
from torch import nn
import sparseconvnet as scn
from torchplus.nn import Empty, Sequential
from torchplus.tools import change_default_args
from second.protos import pipeline_pb2
from google.protobuf import text_format
from second.builder import voxel_builder

class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        # [10+1, 400, 352]
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print('sparse_shape', sparse_shape) # [11, H, W]
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        # [10, 400, 352]
        self.voxel_output_shape = output_shape
        middle_layers = []

        # [128] + [64]
        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        middle_layer1 = []
        for i, o in filters_pairs_d1:
            print(f'middle_layer1 {i} -> {o}')
            middle_layer1.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layer1.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layer1.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], filter_size=(3, 1, 1), filter_stride=(2, 1, 1),
                bias=False))
        middle_layer1.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))

        # assert len(num_filters_down2) > 0
        # [64] + [64, 64]
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        middle_layer2 = []
        for i, o in filters_pairs_d2:
            print(f'middle_layer2 {i} -> {o}')
            middle_layer2.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layer2.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layer2.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], filter_size=(3, 1, 1), filter_stride=(2, 1, 1),
                bias=False))
        middle_layer2.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        #
        self.middle_conv1 = Sequential(*middle_layer1)
        self.middle_conv2 = Sequential(*middle_layer2)
        middle_layers.append(scn.SparseToDense(3, num_filters[-1]))
        self.middle_conv = Sequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        # N, D, H, W -> D, H, W, N
        coors = coors.int()[:, [1, 2, 3, 0]]
        #
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))
        ret = self.middle_conv1(ret)
        print('after conv1', ret) # [2, 64, 5, 400, 352]
        ret = self.middle_conv2(ret)
        print('after conv2', ret) # [2, 64, 2, 400, 352]
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(config_path):
    #
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    #
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)

    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    grid_size = voxel_generator.grid_size
    # [1] + [10, 400, 352] + [128]
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    # [1, 10, 400, 352, 128]
    print('dense_shape', dense_shape)
    middle_num_filters_d1=list(model_cfg.middle_feature_extractor.num_filters_down1)
    middle_num_filters_d2=list(model_cfg.middle_feature_extractor.num_filters_down2)
    middle_feature_extractor = SparseMiddleExtractor(
            output_shape=dense_shape,
            use_norm=True,
            num_input_features=vfe_num_filters[-1],
            num_filters_down1=middle_num_filters_d1,
            num_filters_down2=middle_num_filters_d2)
    middle_feature_extractor = middle_feature_extractor.cuda()
    print(count_parameters(middle_feature_extractor)) # 0.4M

    coors = [[0, 11, 12, 13], [1, 22, 23, 24], [0, 33, 34, 35]]
    coors = torch.Tensor(coors)
    voxel_features = torch.randn(3, vfe_num_filters[-1]).cuda()
    batch_size = 2
    ret = middle_feature_extractor(voxel_features, coors, batch_size)
    print(ret.shape)  # [2, 128, 400, 352]

if __name__ == '__main__':
    fire.Fire()
