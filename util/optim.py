import torch

def decay_weights(optimizer, weight_decay):
    if weight_decay > 0:
        for group in optimizer.param_groups:
            for param in group['params']:
                if len(param.data.shape) > 0:
                    param.data = param.data.add(-weight_decay * group['lr'], param.data)


class TransformsTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors, transforms=None, flatten=True, device=None):
        super(TransformsTensorDataset, self).__init__(*tensors)
        self.transforms = transforms
        self.device = device
        self.flatten=flatten
        
    def __getitem__(self, index):
        X, label = super(TransformsTensorDataset, self).__getitem__(index)
        if self.transforms is not None:
            X = self.transforms(X)
        X = X.reshape((-1,)) if self.flatten else X
        return (X, label)