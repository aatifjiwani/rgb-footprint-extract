import wandb


def wandb_segmentation_image(input, pred_mask, gt_mask, class_labels):
    return wandb.Image(input, masks={
        'predictions': {
            "mask_data": pred_mask,
            "class_labels": class_labels
        },
        'ground_truth': {
            "mask_data": gt_mask,
            "class_labels": class_labels
        }
    })
