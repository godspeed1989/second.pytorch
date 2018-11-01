import torch
import torch.nn as nn

class RCF_RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RCF_RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        # down
        self.conv1_1 = nn.Conv2d(num_input_filters, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        hchn = 128   # hybrid_channels
        self.conv3_1_dsn = nn.Conv2d(256, hchn, 1, padding=0)
        self.conv3_2_dsn = nn.Conv2d(256, hchn, 1, padding=0)
        self.conv3_3_dsn = nn.Conv2d(256, hchn, 1, padding=0)

        self.conv2_1_dsn = nn.Conv2d(128, hchn, 1, padding=0)
        self.conv2_2_dsn = nn.Conv2d(128, hchn, 1, padding=0)

        self.score_dsn2 = nn.Conv2d(hchn, hchn, 3, padding=1)
        self.score_dsn3 = nn.Conv2d(hchn, hchn, 3, padding=1)

        # up
        self.conv3_1_up = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv3_2_up = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_up = nn.Conv2d(256, 128, 3, padding=1)

        self.conv2_1_up = nn.ConvTranspose2d(128+hchn, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv2_2_up = nn.Conv2d(128, 128, 3, padding=1)

        self.conv1_1_up = nn.Conv2d(128+hchn, 128, 3, padding=1)
        self.conv1_2_up = nn.Conv2d(128, 128, 3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # output
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        num_upsample_filters = 128
        self.conv_cls = nn.Conv2d(num_upsample_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(num_upsample_filters, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d( num_upsample_filters, num_anchor_per_loc * 2, 1)

    def forward(self, x):
        # down
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1   = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2   = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3   = self.maxpool(conv3_3)

        conv2_1_dsn = self.conv2_1_dsn(conv2_1)
        conv2_2_dsn = self.conv2_2_dsn(conv2_2)
        conv3_1_dsn = self.conv3_1_dsn(conv3_1)
        conv3_2_dsn = self.conv3_2_dsn(conv3_2)
        conv3_3_dsn = self.conv3_3_dsn(conv3_3)

        so2_out = self.score_dsn2(conv2_1_dsn + conv2_2_dsn)
        so3_out = self.score_dsn3(conv3_1_dsn + conv3_2_dsn + conv3_3_dsn)

        conv3_1_up = self.relu(self.conv3_1_up(pool3))
        conv3_2_up = self.relu(self.conv3_2_up(conv3_1_up))
        conv3_3_up = self.relu(self.conv3_3_up(conv3_2_up))
        conv3_up_cat = torch.cat((conv3_3_up, so3_out), dim=1)

        conv2_1_up = self.relu(self.conv2_1_up(conv3_up_cat))
        conv2_2_up = self.relu(self.conv2_2_up(conv2_1_up))
        conv2_up_cat = torch.cat((conv2_2_up, so2_out), dim=1)

        conv1_1_up = self.relu(self.conv1_1_up(conv2_up_cat))
        conv1_2_up = self.relu(self.conv1_2_up(conv1_1_up))

        x = conv1_2_up

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict
