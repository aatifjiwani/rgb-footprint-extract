import numpy as np

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





    




