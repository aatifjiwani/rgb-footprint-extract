import numpy as np
import models.utils.postprocessing as postprocessing


# Parameters for new metrics
SEPARATION_BUFFER = 2
SMALL_AREA_THRESHOLD = 15
LARGE_AREA_THRESHOLD = 120
ROAD_BUFFER = -5
SMALL_BUILDING_BUFFERS = [1, 3, 5, 8, 10]


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def pixelAcc_manual(self, gt_image, pred_image):
        len_batch = len(gt_image)
        classes = np.unique(gt_image)

        sum_overlap = 0
        sum_n_gt = 0

        for c in classes:
            curr_gt, curr_pred = (gt_image == c).astype(np.int32), (pred_image == c).astype(np.int32)

            overlap = np.sum(np.logical_and(curr_gt, curr_pred))
            n_gt = np.sum(curr_gt)

            sum_overlap += overlap
            sum_n_gt += n_gt

        pixel_acc = sum_overlap / sum_n_gt
        return pixel_acc

    def mIOU_manual(self, gt_image, pred_image):
        len_batch = len(gt_image)
        classes = np.unique(gt_image)

        mIOU = []

        """

        IOU = overlap / union 
            union = (gt + pred - overlap)

        GT       CLASS 0 
                c = 0

        0 0 0
        0 0 0
        0 0 1

        1 1 1  1 1 0
        1 1 1  0 0 0 
        1 1 0  1 1 0

        """

        for c in classes:
            curr_gt, curr_pred = (gt_image == c).astype(np.int32), (pred_image == c).astype(np.int32)

            overlap = np.sum(np.logical_and(curr_gt, curr_pred))
            n_gt = np.sum(curr_gt)
            n_pred = np.sum(curr_pred)

            iou = (overlap) / (n_gt + n_pred - overlap)

            mIOU.append( iou )

        mIOU = sum(mIOU) / len(classes)
        return mIOU

    def f1score_manual(self, gt_image, pred_image):
        num_images = len(gt_image)

        if np.sum(gt_image) == 0:
            return None

        tp = np.sum(pred_image * gt_image)
        fp = np.sum(pred_image*(gt_image==0))
        fn = np.sum((pred_image==0)*gt_image)

        precision = (tp/(tp+fp))
        recall = (tp/(tp+fn))


        numerator = 2*precision * recall
        denominator = precision + recall + 1e-7

        return numerator / denominator

    def f1score_manual_full(self, gt_image, pred_image):
        num_images = len(gt_image)

        if np.sum(gt_image) == 0:
            return np.nan, np.nan, np.nan

        tp = np.sum(pred_image * gt_image)
        fp = np.sum(pred_image*(gt_image==0))
        fn = np.sum((pred_image==0)*gt_image)

        precision = (tp/(tp+fp))
        recall = (tp/(tp+fn))

        if (precision == float('inf') or recall == float('inf')):
            return np.nan, np.nan, np.nan
        elif(np.isnan(precision) or np.isnan(recall)):
            return np.nan, np.nan, np.nan

        numerator = 2*precision * recall
        denominator = precision + recall + 1e-7

        return (numerator / denominator), precision, recall

    def SmIOU(self, gt_image, pred_image, file_name,
              pad_buffers, buffer_val, small_area_thresh, large_area_thresh,
              road_buffer):
        """
        Computes mIoU on small buildings, Small mIoU V1 and Small mIoU V2 for
        each of the paddings in pad_buffers.
        :param gt_image: (np.array)
        :param pred_image: (np.array)
        :param file_name: (str)
        :param pad_buffers: (list of int) small building padding buffers
        :param buffer_val: (int) buffer for small building separation
        :param small_area_thresh: (int) Minimum area for small buildings
        :param large_area_thresh: (int) Maximum area for small buildings
        :param road_buffer: (int) Buffer to filter road predictions
        :return: (dict) with keys [SmIoU-V1, SmIoU-V2-P, mIoU-SB-P]
        """

        # Get gt and predictions for each class
        bg_gt, bg_pred = (gt_image == 0).astype(np.int32), (pred_image == 0).astype(np.int32)
        build_gt, build_pred = (gt_image == 1).astype(np.int32), (pred_image == 1).astype(np.int32)

        # Generate IoU masks for each image in batch
        gt_mask_bg_dict, pred_mask_bg_dict, gt_mask_bd_dict, pred_mask_bd_dict = postprocessing.generate_metric_mask(
            bg_gt=bg_gt, bg_pred=bg_pred, build_gt=build_gt, build_pred=build_pred,
            file_name=file_name, pad_buffers=pad_buffers, buffer_val=buffer_val,
            small_area_thresh=small_area_thresh, large_area_thresh=large_area_thresh,
            road_buffer=road_buffer)

        # Compute mIoU across batch
        smiou_metrics = ['mIoU-SB'] + ['SmIoU-V1-{}'.format(b) for b in pad_buffers] + \
                        ['SmIoU-V2-{}'.format(b) for b in pad_buffers]
        smiou_dict = {}

        for smiou_metric in smiou_metrics:
            if smiou_metric == 'mIoU-SB':
                bg_query, bd_query = 'mIoU-SB', 'mIoU-SB'
            else:
                _, version, pad_buffer = smiou_metric.split('-')
                bg_query = 'SmIoU-{}'.format(pad_buffer)
                bd_query = 'SmIoU-{}'.format(version)

            # * Get masks
            gt_mask_bg = gt_mask_bg_dict[bg_query]
            pred_mask_bg = pred_mask_bg_dict[bg_query]
            gt_mask_bd = gt_mask_bd_dict[bd_query]
            pred_mask_bd = pred_mask_bd_dict[bd_query]

            # * Background
            overlap_bg = np.sum(np.logical_and(gt_mask_bg, pred_mask_bg))
            n_gt_bg = np.sum(gt_mask_bg)
            n_pred_bg = np.sum(pred_mask_bg)
            iou_bg = overlap_bg / (n_gt_bg + n_pred_bg - overlap_bg)

            # * Foreground
            overlap_bd = np.sum(np.logical_and(gt_mask_bd, pred_mask_bd))
            n_gt_bd = np.sum(gt_mask_bd)
            n_pred_bd = np.sum(pred_mask_bd)
            iou_bd = overlap_bd / (n_gt_bd + n_pred_bd - overlap_bd)

            smiou_dict[smiou_metric] = (iou_bg + iou_bd) / 2

        return smiou_dict
