import fire
import torch
from second.protos import pipeline_pb2
from second.pytorch.models.voxelnet_new import RPN
from google.protobuf import text_format
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder

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
    #
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    #
    num_rpn_input_filters = 64
    rpn = RPN(
            use_norm=True,
            num_class=model_cfg.num_class,
            layer_nums=list(model_cfg.rpn.layer_nums),
            layer_strides=list(model_cfg.rpn.layer_strides),
            num_filters=list(model_cfg.rpn.num_filters),
            upsample_strides=list(model_cfg.rpn.upsample_strides),
            num_upsample_filters=list(model_cfg.rpn.num_upsample_filters),
            num_input_filters=num_rpn_input_filters * 2,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=model_cfg.encode_background_as_zeros,
            use_direction_classifier=model_cfg.use_direction_classifier,
            use_bev=model_cfg.use_bev,
            num_groups=model_cfg.rpn.num_groups,
            use_groupnorm=model_cfg.rpn.use_groupnorm,
            box_code_size=target_assigner.box_coder.code_size)
    print(count_parameters(rpn)) # 5M
    spatial_features = torch.randn(1, num_rpn_input_filters * 2, 400, 768)
    spatial_features = spatial_features.cuda()
    rpn = rpn.cuda()
    # spatial_features [Batch, C, H, W]
    preds_dict = rpn(spatial_features)
    # box_preds [Batch, H/2, W/2, 14]
    box_preds = preds_dict["box_preds"]
    print(box_preds.shape)
    # cls_preds [Batch, H/2, W/2, 2]
    cls_preds = preds_dict["cls_preds"]
    print(cls_preds.shape)

if __name__ == '__main__':
    fire.Fire()
