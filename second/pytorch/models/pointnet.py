import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# copy from voxelnet.py
def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PointnetFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=None,
                 num_filters=None,
                 with_distance=None,
                 name='PointnetFeatureExtractor'):
        super(PointnetFeatureExtractor, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

        self.conv1 = torch.nn.Conv1d(3, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64+3, 96, 1)
        self.conv4 = torch.nn.Conv1d(96, 127, 1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(96)
        self.bn4 = torch.nn.BatchNorm1d(127)

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean

        # (K, N, 1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = mask.unsqueeze(-1).type_as(features)

        x = features_relative.permute(0, 2, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)

        x = torch.cat([x, features_relative], dim=-1)
        x *= mask

        x = x.permute(0, 2, 1)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = x.permute(0, 2, 1)

        voxelwise = x.max(dim=1, keepdim=False)[0]
        if self.num_input_features == 4:
            intensity = features[..., 3]
            max_intensity = intensity.max(dim=1, keepdim=True)[0]
            voxelwise = torch.cat([voxelwise, max_intensity], dim=-1)
        return voxelwise


if __name__ == '__main__':
    num_voxel_size = 100
    concated_num_points = 8
    sim_features = Variable(torch.rand(concated_num_points, num_voxel_size, 4))
    sim_num_voxels = Variable(torch.rand(concated_num_points))
    pointfeat = PointnetFeatureExtractor()
    out = pointfeat(sim_features, sim_num_voxels)
    print('global feat', out.size())
