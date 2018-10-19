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

class PointNetfeat(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(PointNetfeat, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, x, num_voxels):
        # x: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        voxel_count = x.shape[1]
        intensity = x[..., 3]
        xyz = x[..., :3]

        features = self.bn1(F.relu(self.conv1(xyz)))
        features = self.bn2(F.relu(self.conv2(features)))
        features = self.bn3(F.relu(self.conv3(features)))
        features = x

        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)

        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


if __name__ == '__main__':
    num_voxel_size = 100
    concated_num_points = 250
    sim_data = Variable(torch.rand(concated_num_points, num_voxel_size, 4))
    pointfeat = PointNetfeat()
    out = pointfeat(sim_data, concated_num_points)
    print('global feat', out.size())
