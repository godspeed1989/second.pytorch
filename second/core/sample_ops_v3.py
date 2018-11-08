import pathlib
import numpy as np
from functools import partial, reduce

from second.core import preprocess as prep
from second.core import box_np_ops
import copy

from second.utils.check import shape_mergeable

"""
    Estimate ground points
"""
class ground_filter(object):
    def __init__(self):
        self.sensor_height = 2.5
        self.max_iter = 15
        self.th_dist = 0.1
        self.stable_delta = 50
        self.reset()

    def reset(self):
        self.normal = None
        self.ground_pc = None
        self.pc = None

    def _extract_initial_seeds(self):
        # LPR is the mean of low point representative
        # lowest 1% points
        sum_cnt = (int)(self.pc.shape[0] * 0.01)
        lpr_height = np.mean(self.pc[:sum_cnt, 2])
        g_seeds_pc = self.pc[self.pc[:, 2] < lpr_height]
        return g_seeds_pc

    def _estimate_plane(self):
        # Create covarian matrix.
        # 1. calculate (x,y,z) mean (3,1)
        mean_xyz = np.expand_dims(np.mean(self.ground_pc, axis=0), axis=-1)
        # 2. calculate covariance
        # cov(x,x), cov(y,y), cov(z,z)
        # cov(x,y), cov(x,z), cov(y,z)
        cov = np.cov(self.ground_pc.T)
        # Singular Value Decomposition: SVD
        u, s, vh = np.linalg.svd(cov)
        # use the least singular vector as normal (3,1)
        self.normal = np.expand_dims(u[:,2], axis=-1)
        # according to normal.T*[x,y,z] = -d
        d_ = -np.dot(self.normal.T, mean_xyz)[0,0]
        # set distance threhold to `th_dist - d`
        th_dist_d_ = - d_ + self.th_dist
        return th_dist_d_

    def filter(self, pc_velo):
        self.pc = pc_velo[:, :3].copy()
        # Error point removal
        # As there are some error mirror reflection under the ground
        # here regardless point under 2* sensor_height
        self.pc = self.pc[self.pc[:, 2] > -1.5* self.sensor_height, :]
        # Sort along Z-axis
        self.pc = self.pc[self.pc[:, 2].argsort()]
        # Extract init ground seeds.
        g_seeds_pc = self._extract_initial_seeds()
        self.ground_pc = g_seeds_pc
        # Ground plane fitter mainloop
        for _ in range(self.max_iter):
            cnt_prev = self.ground_pc.shape[0]
            th_dist_d_ = self._estimate_plane()
            # ground plane model (n,3)*(3,1)=(n,1)
            result = np.dot(self.pc, self.normal)
            result = np.squeeze(result)
            # threshold filter
            sel = result <= th_dist_d_
            self.ground_pc = self.pc[sel, :]
            #print(self.pc.shape[0], ':', cnt_prev, '->', np.sum(sel))
            cnt_delta = abs(np.sum(sel) - cnt_prev)
            if cnt_delta < self.stable_delta:
                break

'''
Input:
    points (N,2)
    voxel_size (2,)  [0.2, 0.2]
    coord_range (2,) [0, -40]  lower bound of coordinate
    grid_size (2,)   [352, 400]
Return;
    voxel_index (M,2)
'''
def _points_to_voxelidx_2d(points, voxel_size, coord_range, grid_size):
    voxel_index = np.floor((points-coord_range) / voxel_size).astype(np.int)
    bound_x = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_box = np.logical_and(bound_x, bound_y)
    voxel_index = voxel_index[bound_box]
    if voxel_index.shape[0] < 2:
        return voxel_index
    else:
        return np.unique(voxel_index, axis=0)

'''
Input: xyz -> xy
    points (N,3)
    voxel_size (3,)  [0.2, 0.2, 0.4]
    coord_range (3,) [0, -40, -3]  lower bound of coordinate
    grid_size (3,)   [352, 400,  10]
Return;
    grid_mask (grid_size[0], grid_size[1])
'''
def _points_to_gridmask_2d(points, voxel_size, coord_range, grid_size):
    grid_mask = np.zeros(grid_size[:2], dtype=np.bool_)
    #
    voxel_index = _points_to_voxelidx_2d(points[:,:2], voxel_size[:2], coord_range[:2], grid_size[:2])
    # idx from (N,2) -> (2,N)
    grid_mask[tuple(voxel_index.T)] = True
    #
    return grid_mask

def _fine_sample_by_grd(sampled_cls, grd_gridmask, voxel_size, coord_range, grid_size):
    new_sampled_cls = []
    for s in sampled_cls:
        boxes = s['box3d_lidar'][np.newaxis, ...]
        boxes_bv = box_np_ops.center_to_corner_box2d(
            boxes[:, 0:2], boxes[:, 3:5], boxes[:, 6])
        boxes_bv = np.reshape(boxes_bv, [-1, 2])
        voxel_idx = _points_to_voxelidx_2d(boxes_bv, voxel_size[:2], coord_range[:2], grid_size[:2])
        if np.all(grd_gridmask[tuple(voxel_idx.T)]):
            new_sampled_cls.append(s)
    #print(len(sampled_cls), '->', len(new_sampled_cls))
    return new_sampled_cls


'''
    from second.core.sample_ops import DataBaseSamplerV2
'''
class DataBaseSamplerV3:
    def __init__(self, db_infos, groups, db_prepor=None,
                 rate=1.0, global_rot_range=None):
        for k, v in db_infos.items():
            print(f"load {len(v)} {k} database infos")

        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            print("After filter database:")
            for k, v in db_infos.items():
                print(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = False  # slower
        if any([len(g) > 1 for g in groups]):
            self._use_group_sampling = True
        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos  # just use db_infos
            for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
        else:
            for group_info in groups:
                group_dict = {}
                group_names = list(group_info.keys())
                group_name = ", ".join(group_names)
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
                self._group_name_to_names.append((group_name, group_names))
                # self._group_name_to_names[group_name] = group_names
                for name in group_names:
                    for item in db_infos[name]:
                        gid = item["group_id"]
                        if gid not in group_dict:
                            group_dict[gid] = [item]
                        else:
                            group_dict[gid] += [item]
                if group_name in self._group_db_infos:
                    raise ValueError("group must be unique")
                group_data = list(group_dict.values())
                self._group_db_infos[group_name] = group_data
                info_dict = {}
                if len(group_info) > 1:
                    for group in group_data:
                        names = [item["name"] for item in group]
                        names = sorted(names)
                        group_name = ", ".join(names)
                        if group_name in info_dict:
                            info_dict[group_name] += 1
                        else:
                            info_dict[group_name] = 1
                print(info_dict)


        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = prep.BatchSampler(v, k)
        self._enable_global_rot = False
        if global_rot_range is not None:
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            if np.abs(global_rot_range[0] -
                        global_rot_range[1]) >= 1e-3:
                self._enable_global_rot = True
        self._global_rot_range = global_rot_range

    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    def sample_all(self,
                   points,
                   voxel_size,
                   point_cloud_range,
                   voxel_grids,
                   root_path,
                   gt_boxes,
                   gt_names,
                   num_point_features,
                   random_crop=False,
                   gt_group_ids=None,
                   rect=None,
                   Trv2c=None,
                   P2=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self._sample_classes,
                                              self._sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        finetune_by_grd = True
        voxel_scale = 2
        voxel_size_scaled = voxel_size * voxel_scale
        voxel_grids_scale = voxel_grids // voxel_scale
        if finetune_by_grd:
            grd_filter = ground_filter()
            grd_filter.filter(points)
            grd_gridmask = _points_to_gridmask_2d(grd_filter.ground_pc,
                                                  voxel_size_scaled, point_cloud_range[:3], voxel_grids_scale)
        else:
            grd_gridmask = None

        for class_name, sampled_num in zip(sampled_groups,
                                           sample_num_per_class):
            if sampled_num > 0:
                if finetune_by_grd:
                    assert self._use_group_sampling is not True
                    all_samples = self._sampler_dict[class_name].sample(sampled_num * 25)
                    all_samples = copy.deepcopy(all_samples)
                    all_samples = _fine_sample_by_grd(all_samples, grd_gridmask,
                                                      voxel_size_scaled, point_cloud_range[:3], voxel_grids_scale)
                    if len(all_samples) > sampled_num:
                        #print(len(all_samples), '>', sampled_num)
                        all_samples = all_samples[:sampled_num]
                else:
                    all_samples = self._sampler_dict[class_name].sample(sampled_num)
                    all_samples = copy.deepcopy(all_samples)

                if self._use_group_sampling:
                    sampled_cls = self.sample_group(class_name, sampled_num,
                                                    avoid_coll_boxes, total_group_ids)
                else:
                    if len(all_samples) > 0:
                        sampled_cls = self.sample_class_v2(all_samples,
                                                           avoid_coll_boxes)
                    else:
                        sampled_cls = []

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0)

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)
                    if self._use_group_sampling:
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(sampled_cls[0]["group_id"])[np.newaxis, ...]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0)
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0)

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            num_sampled = len(sampled)
            s_points_list = []
            for info in sampled:
                s_points = np.fromfile(
                    str(pathlib.Path(root_path) / info["path"]),
                    dtype=np.float32)
                s_points = s_points.reshape([-1, num_point_features])
                # if not add_rgb_to_points:
                #     s_points = s_points[:, :4]
                if "rot_transform" in info:
                    rot = info["rot_transform"]
                    s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                        s_points[:, :3], rot, axis=2)
                s_points[:, :3] += info["box3d_lidar"][:3]
                s_points_list.append(s_points)
                # print(pathlib.Path(info["path"]).stem)
            # gt_bboxes = np.stack([s["bbox"] for s in sampled], axis=0)
            # if np.random.choice([False, True], replace=False, p=[0.3, 0.7]):
            # do random crop.
            if random_crop:
                s_points_list_new = []
                gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect,
                                                     Trv2c, P2)
                crop_frustums = prep.random_crop_frustum(
                    gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = prep.mask_points_in_corners(
                        s_points, crop_frustums[i:i + 1]).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (
                            s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new

            finetune_samples_axis_z = True
            if finetune_samples_axis_z:
                gt_box_bottom_avg = np.mean(gt_boxes[:,2])
                # finetune boxes
                sampled_gt_boxes_bottom = sampled_gt_boxes[:,2]
                delta = sampled_gt_boxes_bottom - gt_box_bottom_avg
                sampled_gt_boxes[:,2] -= delta
                # finetune points
                for i in range(len(s_points_list)):
                    s_points_list[i][:,2] -= delta[i]

            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled, ), dtype=np.bool_)
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled))
        else:
            ret = None
        return ret

    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret), ), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            group_name = ", ".join(name)
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret), ), dtype=np.int64)

    def sample_class_v2(self, sampled, gt_boxes):
        # sample some from dict
        # sampled = self._sampler_dict[name].sample(num)
        # sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        # BEV: ground truth
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        # BEV: sampled
        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        # just rotate sampled objects
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask,
             np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                num_try=100)
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, -1] - sp_boxes[i - num_gt, -1])
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)
        # rewrite sampled group id to avoid duplicated with gt group ids
        gid_map = {}
        max_gt_gid = np.max(gt_group_ids)
        sampled_gid = max_gt_gid + 1
        for s in sampled:
            gid = s["group_id"]
            if gid in gid_map:
                s["group_id"] = gid_map[gid]
            else:
                gid_map[gid] = sampled_gid
                s["group_id"] = sampled_gid
                sampled_gid += 1

        num_gt = gt_boxes.shape[0]
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_group_ids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask,
             np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100)
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        valid_samples = []
        idx = num_gt
        for num in group_num:
            if coll_mat[idx:idx + num].any():
                coll_mat[idx:idx + num] = False
                coll_mat[:, idx:idx + num] = False
            else:
                for i in range(num):
                    if self._enable_global_rot:
                        sampled[idx - num_gt + i]["box3d_lidar"][:2] = boxes[idx + i, :2]
                        sampled[idx - num_gt + i]["box3d_lidar"][-1] = boxes[idx + i, -1]
                        sampled[idx - num_gt + i]["rot_transform"] = (
                            boxes[idx + i, -1] - sp_boxes[idx + i - num_gt, -1])

                    valid_samples.append(sampled[idx - num_gt + i])
            idx += num
        return valid_samples