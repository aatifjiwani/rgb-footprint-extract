import os
import torch
from models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback

def load_model(model, resume_dataset=None, best_miou=False, is_cuda=False, gpu_ids=None):
    # Load state_dict, if any
    model_checkpoint = None
    if resume_dataset is not None:
        checkpoint_name = "best_miou_checkpoint.pth.tar" if best_miou else "best_loss_checkpoint.pth.tar" 
        checkpoint_path = os.path.join("weights", resume_dataset, checkpoint_name)
        print("Resuming from {}".format(checkpoint_path))

        model_checkpoint = torch.load(checkpoint_path)  

    # Load model onto GPUs
    if is_cuda:
        assert gpu_ids is not None
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()


        if model_checkpoint is not None:
            model.load_state_dict(model_checkpoint)
    elif model_checkpoint is not None:
        try:
            model.load_state_dict(model_checkpoint)
        except RuntimeError:
            # The model is currently on the CPU, and does not have DataParallel wrapper
            # Need to remove the "module." prefix from all keys in state_dict
            from collections import OrderedDict
            new_checkpoint = OrderedDict()
            for module_name, parameters in model_checkpoint.items():
                name = module_name[7:] 
                new_checkpoint[name] = parameters
            model_checkpoint = new_checkpoint
            model.load_state_dict(model_checkpoint)
        
    return model
