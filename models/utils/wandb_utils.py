import wandb


def wandb_segmentation_image(input_img, pred_mask, gt_mask, class_labels):
    print(input_img.shape)
    print(pred_mask.shape)
    print(gt_mask.shape)
    return wandb.Image(input_img, masks={
        'predictions': {
            "mask_data": pred_mask,
            "class_labels": class_labels
        },
        'ground_truth': {
            "mask_data": gt_mask,
            "class_labels": class_labels
        }
    })
