"""Region Proposal Network"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Tuple, Optional
from torch.functional import Tensor
from torchvision.ops import boxes as box_ops
from _utils import BalancedPositiveNegativeSampler, Matcher

from collections import OrderedDict
from fpn import FeaturePyramidNetwork
"""
Args:
images taking shape b, c, h, w
image sizes if from images using [image.shape[-2:] for image in images]
"""
class ImageList:
    def __init__(self, images:Tensor, image_sizes:List[Tuple[int, int]]):
        self.images = images
        self.image_sizes = image_sizes

@torch.jit.script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights:Tuple[float, float, float, float]=(1.0, 1.0, 1.0, 1.0), bbx_clip:float=math.log(1000.0 / 16) ) -> None:
        self.weights = weights
        self.bbx_clip = bbx_clip

    def decode(self, pred_bbx_deltas:Tensor, anchors:List[Tensor]):
        boxes_per_image  = [ a.size(0) for a in anchors ]
        boxes_sum = sum(boxes_per_image)
        concat_anchors = torch.cat(anchors, dim=0)
        assert boxes_sum > 0, "Sum of pred boxes per image in decode < 0"
        pred_bbx = pred_bbx_deltas.reshape(boxes_sum, -1)
        pred_bbx = self.decode_single(pred_bbx, concat_anchors)
        return pred_bbx.reshape(boxes_sum, -1, 4)


    def decode_single(self, pred_bbx_deltas:Tensor, anchors:Tensor) -> Tensor:
        """
        Notes:
        0 or 0::4 = x1, 1 or 1::4 = y1, 2 or 2::4 =x2, 3 or 3::4 =y2
        """
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights

        dx = pred_bbx_deltas[:, 0::4] / wx
        dy = pred_bbx_deltas[:, 1::4] / wy
        dw = pred_bbx_deltas[:, 2::4] / ww
        dh = pred_bbx_deltas[:, 3::4] / wh

        #Avoid sending large values to exp
        dw = dw.clamp(max=self.bbx_clip)
        dh = dh.clamp(max=self.bbx_clip)

        pred_cx = dx * widths[:, None] + cx[:, None]
        pred_cy = dy * heights[:, None] + cy[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_cx.dtype) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_cy.dtype) * pred_w

        pred_boxes1 = pred_cx - c_to_c_w
        pred_boxes2 = pred_cy - c_to_c_h
        pred_boxes3 = pred_cx + c_to_c_w
        pred_boxes4 = pred_cy + c_to_c_h
        pred_boxes = torch.stack([pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4], dim=2).flatten(1)

        return pred_boxes

    def encode(self, matched_gt_boxes:List[Tensor], proposed_boxes:List[Tensor]) -> List[Tensor]:
        boxes_per_image = [ len(box) for box in proposed_boxes ]
        reference_boxes = torch.cat(matched_gt_boxes, dim=0)
        proposals = torch.cat(proposed_boxes, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, dim=0)

    def encode_single(self, reference_boxes:Tensor, proposals:Tensor) -> Tensor:
        device = reference_boxes.device
        dtype = reference_boxes.dtype
        weights = torch.as_tensor(self.weights, device=device, dtype=dtype)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets


"""
Args:
sizes: ((64,), (128,), (256,))
aspect_ratios: [0.5, 1, 2] * len(sizes) so that each size has an aspect_ratio

@function generate_anchors
[0.5, 1, 2] * 64 = [[32, 65, 128]] and view(-1) = [32, 64, 128]
same applies for **height

-32, -32, 32, 32
-64, -64, 64, 64
-128, -128, 128, 128

the stack is divided by 2 to preseve the original intended area
above are coordinates for 1 size and 3 different aspect ratios

result of cell_anchors = List[Tensor] = len(sizes) where each cell_anchor has shape len(aspect_ratio) * 4

@function generate_grid_anchors
Note generating grid anchor boxes is w.r.t the entire image

Returns anchors_for_images : List[Tensor] where each element inside the list is
a tensor containing bounding boxes for an image's  feature maps
"""
class AnchorGenerator(nn.Module):
    def __init__(self, sizes:Tuple[Tuple[int], ...], aspect_ratios:Tuple[Tuple[float, ...], ...]):
        super(AnchorGenerator, self).__init__()
        assert len(sizes) == len(aspect_ratios)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [ self.generate_anchors(s, a) for s, a in zip(sizes, aspect_ratios)]

    def get_anchors_per_cell(self, ) -> int:
        return [ len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios) ][0]

    def generate_anchors(self, size:Tuple[int], aspect_ratios:Tuple[float, ...]) -> Tensor:
        area = torch.Tensor(size)
        ratios = torch.Tensor(aspect_ratios)
        width_ratios = torch.sqrt(ratios)
        height_ratios = 1 / width_ratios
        widths = (width_ratios[:, None] * area[None, :]).view(-1)
        heights = (height_ratios[:, None] * area[None, :]).view(-1)
        base_anchors = torch.stack([-widths, -heights, widths, heights], dim=1) / 2
        return base_anchors.round()

    def generate_grid_anchors(self, grid_sizes, strides:List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        assert len(grid_sizes) == len(strides) == len(self.cell_anchors), "Size of cell anchors should be equal to feature map layers"
        anchors = []
        for (grid_size, stride, base_anchor) in zip(grid_sizes, strides, self.cell_anchors):
            grid_h, grid_w = grid_size
            stride_h, stride_w = stride
            shifts_x = torch.arange(0, grid_w) * stride_w
            shifts_y = torch.arange(0, grid_h) * stride_h
            shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_x, indexing='ij')
            shift_x = shifts_x.reshape(-1)
            shift_y = shifts_y.reshape(-1)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchor.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def forward(self, images: ImageList, feature_maps:List[Tensor] ) -> List[Tensor]:
        grid_sizes = [ feature.shape[-2:]  for feature in feature_maps]
        image_size = images.images.shape[-2:]
        strides = [(
            torch.tensor(image_size[0] // grid_size[0], dtype=torch.int64),
            torch.tensor(image_size[1] // grid_size[1], dtype=torch.int64)
        )
            for grid_size in grid_sizes
        ]

        anchors_for_images = []
        anchors_all_feature_maps = self.generate_grid_anchors(grid_sizes, strides)
        for _ in range(len(images.image_sizes)):
            anchor_in_image = [  anchor_in_feature_map for anchor_in_feature_map in  anchors_all_feature_maps]
            anchor_in_image = torch.cat(anchor_in_image, 0)
            anchors_for_images.append(anchor_in_image)
        return anchors_for_images

"""
Args:
x = Dict = ['f1' to torch.Tensor(2, 256, w, h), 'f2' to torch.Tensor(2, 128, w, h)]

Returns: A tuple where the item 1 in the tuple = List of logits for all feature_maps
and item 2 in the tuple = list of all bbx coords where each item in the list is bbx_coords
tensor for a feature_map
"""
class RegionProposalHead(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_anchors:int):
        super(RegionProposalHead, self).__init__()
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(out_channels, num_anchors, kernel_size=1, stride=1)
        self.bbx_reg = nn.Conv2d(out_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x:Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        feature_maps = list(x.values())
        logits = []
        bbx_regs = []
        for feature_map in feature_maps:
            y = self.conv_x(feature_map)
            logits.append(self.cls_logits(y))
            bbx_regs.append(self.bbx_reg(y))
        return logits, bbx_regs

"""
Args:
x = feature map as Dict containing name of feature_map and tensor
"""
class RegionProposalNetwork(nn.Module):
    def __init__(
        self, in_channels:int, out_channels:int, sizes:Tuple[Tuple[int], ...], aspect_ratios:Tuple[Tuple[float, ...]],
        pre_nms_top_n:Dict[str, int], post_nms_top_n:Dict[str, int], score_threshold:float, nms_theshold:float,
        fg_iou_threshold:float, bg_iou_threshold:float, batch_size_per_image:int, positive_fraction:float, min_box_size:float=1e-3,
    ):
        super(RegionProposalNetwork, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = len(sizes) * aspect_ratios
        self.num_anchors_per_cell = len(aspect_ratios) * len(sizes)
        self.rpn_head = RegionProposalHead(in_channels, out_channels, self.num_anchors_per_cell)
        self.generate_anchors = AnchorGenerator(self.sizes, self.aspect_ratios)
        self.box_coder= BoxCoder()

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.score_threshold = score_threshold
        self.nms_theshold = nms_theshold
        self.min_box_size = min_box_size

        self.proposal_matcher = Matcher(fg_iou_threshold, bg_iou_threshold, False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        #len(aspect_ratio) * len(sizes) = num_anchors
    def permute_and_reshape(self, layer:Tensor, B, C, W, H):
        """
        Notes:
        -1 = number of anchors
        permute -> for each height and width we want all the
        anchors in the cell and for all the anchors we want the class assigned to the cell
        """
        layer = layer.view(B, -1, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        return layer.reshape(B, -1, C)

    def concat_bbx_logit_layers(self, objectness:List[Tensor], bbx_regs:List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Notes:
        This function returns all the bbx_reg coordinates for all images in B
        Note that each image further has bbx coordinates for each feature map
        For example: x, 4 for bbx_reg and x, 2 for bbx_cls
        """
        flattened_bbx_cls = []
        flattened_bbx_regs = []
        for cls_per_level, bbx_reg_per_level in zip(objectness, bbx_regs):
            B, Axc, H, W = cls_per_level.shape
            _, Ax4, _, _ = bbx_reg_per_level.shape
            A = Ax4 // 4
            C = Axc // A
            flattened_bbx_cls.append(self.permute_and_reshape(cls_per_level, B, C, W, H))
            flattened_bbx_regs.append(self.permute_and_reshape(bbx_reg_per_level, B, 4, W, H))


        bbx_cls = torch.cat(flattened_bbx_cls, dim=1).flatten(0, -2)
        bbx_reg = torch.cat(flattened_bbx_regs, dim=1).reshape(-1, 4)
        return bbx_cls, bbx_reg

    def pre_nms_top_n(self, ) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self, ) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]


    def get_top_n_idx(self, objectness:Tensor, num_anchors_per_level:List[int]) -> Tensor:
        top_n_idx = []
        anchor_idx_offset = 0
        for o in  objectness.split(num_anchors_per_level, dim=1):
            anchors_in_level = o.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), anchors_in_level)
            _, anchor_idx = o.topk(pre_nms_top_n, dim=1)
            top_n_idx.append(anchor_idx + anchor_idx_offset)
            anchor_idx_offset += anchors_in_level
        return torch.cat(top_n_idx, dim=1)



    def filter_proposals(self, proposals:Tensor, objectness:Tensor, num_anchors_per_level:List[int], image_shapes:List[Tuple[int,int]]) -> Tuple[List[Tensor], List[Tensor]]:
        final_boxes = []
        final_scores = []

        num_images = proposals.shape[0]
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        #Giving anchors in each level an index indicating the level they belong to
        levels = [ torch.full((anchors_in_level, ), idx, dtype=torch.int64) for idx, anchors_in_level in  enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, dim=0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        top_n_idx = self.get_top_n_idx(objectness, num_anchors_per_level)
        batch_images = torch.arange(0, num_images)
        batch_idx = batch_images[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]

        objectness = torch.sigmoid(objectness)

        for boxes, scores, level, image_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            keep = box_ops.remove_small_boxes(boxes, self.min_box_size)
            boxes, scores, level = boxes[keep], scores[keep], level[keep]

            keep = torch.where(scores >= self.score_threshold)[0]
            boxes, scores, level = boxes[keep], scores[keep], level[keep]

            keep = box_ops.batched_nms(boxes, scores, level, self.nms_theshold)
            keep = keep[:self.post_nms_top_n()]

            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors:List[Tensor], targets:List[Dict[str,Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Notes:
        Purpose is to find the anchor boxes with good overlaps with the target boxes
        box_ops.box_iou(gt, anchors) if gt = N, 4 matrix and anchors = M, 4 matrix then
        this function returns an N, M matrix where the values of the matrix = IoU
        """
        labels = []
        matched_gt_boxes = []
        for anchor_per_image, target_per_image in zip(anchors, targets):
            gt_boxes = target_per_image['boxes']
            if gt_boxes.numel() == 0:
                device = anchor_per_image.device
                matched_gt_boxes_per_img = torch.zeros(anchor_per_image.shape, device=device, dtype=torch.float32)
                labels_per_image = torch.zeros(anchor_per_image.shape[0], device=device, dtype=torch.float32)
            else:
                matched_quality_matrix = box_ops.box_iou(gt_boxes, anchor_per_image)
                matched_idx = self.proposal_matcher(matched_quality_matrix)
                #clamp because some indices may have -ve values
                matched_gt_boxes_per_img = gt_boxes[matched_idx.clamp(min=0)]

                labels_per_image = matched_idx >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                bg_idx = matched_idx == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_idx] = 0.0

                #discard indices between threshholds
                idx_discard = matched_idx == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[idx_discard] = -1.0

                labels.append(labels_per_image)
                matched_gt_boxes.append(matched_gt_boxes_per_img)

        return matched_gt_boxes, labels


    def compute_loss(self, objectness:Tensor, labels:List[Tensor], pred_bbox_deltas:Tensor, regression_targets:List[Tensor]):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        _labels = torch.cat(labels, dim=0)
        reg_targets = torch.cat(regression_targets, dim=0)

        box_loss = (
            F.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                reg_targets[sampled_pos_inds],
                beta=1 / 9,
                reduction="sum",
            )
            / (sampled_inds.numel())
        )

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], _labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(self, images: ImageList, x:Dict[str, Tensor], targets:Optional[List[Dict[str, Tensor]]]=None) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        feature_maps = list(x.values())
        objectness, bbx_regs = self.rpn_head(x)
        anchors_for_images = self.generate_anchors(images, feature_maps)

        num_images = len(anchors_for_images)
        num_anchors_per_level_shape = [ o[0].shape for o in objectness ]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape ]

        objectness, bbx_regs = self.concat_bbx_logit_layers(objectness, bbx_regs)
        proposals = self.box_coder.decode(bbx_regs.detach(), anchors_for_images)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.filter_proposals(proposals, objectness, num_anchors_per_level, images.image_sizes)

        losses = {}
        if self.training:
            print('Training RPN ...')
            assert targets is not None, "During training targets should not be None."
            matched_gt_boxes, labels = self.assign_targets_to_anchors(anchors_for_images, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors_for_images)
            rpn_cls_losses, rpn_bbx_loss = self.compute_loss(objectness, labels, bbx_regs, regression_targets)
            losses = { "rpn_cls_losses": rpn_cls_losses, "rpn_bbx_loss": rpn_bbx_loss }

        return boxes, losses

if __name__ == "__main__":

    #gt_boxes = 2 images, 7 anchors with 4 coordinates
    images, gt_boxes = torch.randn([2, 3, 224, 224]), torch.rand(2, 7, 4)
    image_shapes = [ image.shape[-2:] for image in images ]
    image_list = ImageList(images, image_shapes)

    #2 images, 11 anchors
    labels = torch.randint(0, 4, [2, 7])
    targets = []
    for i in range(len(images)):
        targets_dict = {}
        targets_dict['boxes'] = gt_boxes[i]
        targets_dict['labels'] = labels[i]
        targets.append(targets_dict)

    assert gt_boxes.shape[0] == images.shape[0], "First dim of boxes should be equal to number of images in batch"
    assert labels.shape[0] == images.shape[0], "First dim of labels should be equal to the number of images in batch"

    dict = OrderedDict()
    dict['f1'] = torch.randn(2, 64, 112, 112)
    dict['f2'] = torch.randn(2, 128, 64, 64)
    dict['f3'] = torch.randn(2, 256, 32, 32)

    assert all([ v.shape[0] == images.shape[0] for k, v in dict.items()]), "Batch size for feature_maps != image batch size"

    fpn = FeaturePyramidNetwork([64,128,256], 256)
    feature_maps = fpn(dict)
    print([(k, v.shape) for k, v in feature_maps.items()])

    in_channels = 256
    out_channels = 25
    sizes = ((32, ), (64, ), (128, ))
    aspect_ratios = ((0.5, 1.0, 2.0), )
    pre_nms_top_n = { "training": 40, "testing": 10 }
    post_nms_top_n = { "training": 40, "testing": 10 }
    score_threshold = 0.05
    nms_threshold = 0.7
    fg_iou_threshold = 0.5
    bg_iou_threshold = 0.3
    batch_size_per_image = 256
    positive_fraction = 0.5


    rpn = RegionProposalNetwork(
        in_channels,
        out_channels,
        sizes, aspect_ratios,
        pre_nms_top_n,
        post_nms_top_n,
        score_threshold,
        nms_threshold,
        fg_iou_threshold,
        bg_iou_threshold,
        batch_size_per_image,
        positive_fraction
    )

    boxes, losses = rpn(image_list, feature_maps, targets)
    print(f'Boxes shape {[box.shape for box in boxes ]}')
    if len(losses) > 1:
        print(f'\nLosses: {losses}')





