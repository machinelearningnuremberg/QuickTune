import copy
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import tqdm
import os
from .stoch_norm import StochNorm1d, StochNorm2d, StochNorm3d

torch.set_printoptions(precision=10)
class SPRegularization(nn.Module):
    r"""
    Based on the implementation in: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/delta.py#L35

    The SP (Starting Point) regularization from `Explicit inductive bias for transfer learning with convolutional networks
    (ICML 2018) <https://arxiv.org/abs/1802.01483>`_
    The SP regularization of parameters :math:`w` can be described as:
    .. math::
        {\Omega} (w) = \dfrac{1}{2}  \Vert w-w_0\Vert_2^2 ,
    where :math:`w_0` is the parameter vector of the model pretrained on the source problem, acting as the starting point (SP) in fine-tuning.
    Args:
        source_model (torch.nn.Module):  The source (starting point) model.
        target_model (torch.nn.Module):  The target (fine-tuning) model.
    Shape:
        - Output: scalar.
    """
    def __init__(self, model: nn.Module, head_name, regularization_weight: float = 0.01):
        super(SPRegularization, self).__init__()
        self.target_model = model
        self.source_weight = {}
        self.regularization_weight = regularization_weight
        self.head_name = head_name

        #Only compared with unfrozen weights different form the head
        for name, param in model.named_parameters():
            if (not name.startswith(self.head_name)) and param.requires_grad: # it might not work for every model
                self.source_weight[name] = param.clone().detach()

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            if name in self.source_weight:
                #print("Param:{:10f}".format(param.mean().double()-self.source_weight[name].mean().double()))
                output += self.regularization_weight * torch.norm(param - self.source_weight[name]) ** 2
        return output

class BatchSpectralShrinkage(nn.Module):
    r"""
    Based on the github code: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/bss.py#L11

    The regularization term in `Catastrophic Forgetting Meets Negative Transfer:
    Batch Spectral Shrinkage for Safe Transfer Learning (NIPS 2019) <https://proceedings.neurips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf>`_.
    The BSS regularization of feature matrix :math:`F` can be described as:
    .. math::
        L_{bss}(F) = \sum_{i=1}^{k} \sigma_{-i}^2 ,
    where :math:`k` is the number of singular values to be penalized, :math:`\sigma_{-i}` is the :math:`i`-th smallest singular value of feature matrix :math:`F`.
    All the singular values of feature matrix :math:`F` are computed by `SVD`:
    .. math::
        F = U\Sigma V^T,
    where the main diagonal elements of the singular value matrix :math:`\Sigma` is :math:`[\sigma_1, \sigma_2, ..., \sigma_b]`.
    Args:
        k (int):  The number of singular values to be penalized. Default: 1
    Shape:
        - Input: :math:`(b, |\mathcal{f}|)` where :math:`b` is the batch size and :math:`|\mathcal{f}|` is feature dimension.
        - Output: scalar.
    """
    def __init__(self, k=1, regularization_weight=0.5):
        super(BatchSpectralShrinkage, self).__init__()
        self.k = k
        self.regularization_weight = regularization_weight

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return self.regularization_weight*result



class BehavioralRegularization(nn.Module):
    r"""
    Based on: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/delta.py#L69
    The behavioral regularization from `DELTA:DEep Learning Transfer using Feature Map with Attention
    for convolutional networks (ICLR 2019) <https://openreview.net/pdf?id=rkgbwsAcYm>`_
    It can be described as:
    .. math::
        {\Omega} (w) = \sum_{j=1}^{N}   \Vert FM_j(w, \boldsymbol x)-FM_j(w^0, \boldsymbol x)\Vert_2^2 ,
    where :math:`w^0` is the parameter vector of the model pretrained on the source problem, acting as the starting point (SP) in fine-tuning,
    :math:`FM_j(w, \boldsymbol x)` is feature maps generated from the :math:`j`-th layer of the model parameterized with :math:`w`, given the input :math:`\boldsymbol x`.
    Inputs:
        layer_outputs_source (OrderedDict):  The dictionary for source model, where the keys are layer names and the values are feature maps correspondingly.
        layer_outputs_target (OrderedDict):  The dictionary for target model, where the keys are layer names and the values are feature maps correspondingly.
    Shape:
        - Output: scalar.
    """
    def __init__(self, source_model, regularization_weight=0.01):
        super(BehavioralRegularization, self).__init__()
        self.regularization_weight = regularization_weight
        self.source_model = source_model

        for param in self.source_model.parameters():
            param.requires_grad = False
    def forward(self, layer_outputs_source, layer_outputs_target):
        output = self.regularization_weight*(torch.norm(layer_outputs_target - layer_outputs_source.detach()) ** 2)
        return output

class CoTuningLoss(nn.Module):
    """
    Based on code: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/co_tuning.py

    The Co-Tuning loss in `Co-Tuning for Transfer Learning (NIPS 2020)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf>`_.
    Inputs:
        - input: p(y_s) predicted by source classifier.
        - target: p(y_s|y_t), where y_t is the ground truth class label in target dataset.
    Shape:
        - input:  (b, N_p), where b is the batch size and N_p is the number of classes in source dataset
        - target: (b, N_p), where b is the batch size and N_p is the number of classes in source dataset
        - Outputs: scalar.
    """
    def __init__(self, regularization_weight=0.01):
        super(CoTuningLoss, self).__init__()
        self.regularization_weight = regularization_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = - target * F.log_softmax(input, dim=-1)
        y = torch.mean(torch.sum(y, dim=-1))
        return self.regularization_weight*y


class Relationship(object):
    """
    Based on code: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/co_tuning.py

    Learns the category relationship p(y_s|y_t) between source dataset and target dataset.
    Args:
        data_loader (torch.utils.data.DataLoader): A data loader of target dataset.
        classifier (torch.nn.Module): A classifier for Co-Tuning.
        device (torch.nn.Module): The device to run classifier.
        cache (str, optional): Path to find and save the relationship file.
    """
    def __init__(self, data_loader, classifier, device, cache=None):
        super(Relationship, self).__init__()
        self.data_loader = data_loader
        self.classifier = classifier
        self.device = device
        if cache is None or not os.path.exists(cache):
            source_predictions, target_labels = self.collect_labels()
            self.relationship = self.get_category_relationship(source_predictions, target_labels)
            if cache is not None:
                np.save(cache, self.relationship)
        else:
            self.relationship = np.load(cache)

    def __getitem__(self, category):

        if len(category.shape)>1:
            if len(category[0])>1:
                category = np.argmax(category, axis=1)

        return self.relationship[category]

    def collect_labels(self):
        """
        Collects predictions of target dataset by source model and corresponding ground truth class labels.
        Returns:
            - source_probabilities, [N, N_p], where N_p is the number of classes in source dataset
            - target_labels, [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset
        """

        print("Collecting labels to calculate relationship")
        source_predictions = []
        target_labels = []

        self.classifier.eval()
        with torch.no_grad():
            for i, (x, label) in enumerate(tqdm.tqdm(self.data_loader)):
                x = x.to(self.device)
                y_s, _, _ = self.classifier(x)

                source_predictions.append(F.softmax(y_s, dim=1).detach().cpu().numpy())
                target_labels.append(label.cpu().numpy())

        return np.concatenate(source_predictions, 0), np.concatenate(target_labels, 0)

    def get_category_relationship(self, source_probabilities, target_labels):
        """
        The direct approach of learning category relationship p(y_s | y_t).
        Args:
            source_probabilities (numpy.array): [N, N_p], where N_p is the number of classes in source dataset
            target_labels (numpy.array): [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset
        Returns:
            Conditional probability, [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
        """
        if len(target_labels.shape)>1:
            if len(target_labels[0])>1:
                target_labels = np.argmax(target_labels, axis=1)
        N_t = np.max(target_labels) + 1  # the number of target classes
        conditional = []
        for i in range(N_t):
            this_class = source_probabilities[target_labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)
        return np.concatenate(conditional)

def convert_to_stoch_norm(module, p = 0.5):
    """
    Based on: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/normalization/stochnorm.py#L254

    Traverses the input module and its child recursively and replaces all
    instance of BatchNorm to StochNorm.
    Args:
        module (torch.nn.Module): The input module needs to be convert to StochNorm model.
        p (float): The hyper-parameter for StochNorm layer.
    Returns:
         The module converted to StochNorm version.
    """

    mod = module
    for pth_module, stoch_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                         torch.nn.modules.batchnorm.BatchNorm2d,
                                         torch.nn.modules.batchnorm.BatchNorm3d],
                                        [StochNorm1d,
                                         StochNorm2d,
                                         StochNorm3d]):
        if isinstance(module, pth_module):
            mod = stoch_module(module.num_features, module.eps, module.momentum, module.affine, p)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var

            if module.affine and module.weight.requires_grad:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_to_stoch_norm(child, p))

    return mod