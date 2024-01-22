import copy
import torch
from torch import nn
import timm.models as models
from collections import OrderedDict
import functools
import os
import json
from hpo.utils import find_file
import time

def compute_gradient_norm(module, head_name = None):
    #taken from: https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/3

    if head_name is None:
        total_norm = 0
        parameters = [p for p in module.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            if not torch.isnan(param_norm):
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm, None
    else:
        head_norm = 0
        backbone_norm = 0
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                if name.startswith(head_name):
                    head_norm += param_norm.item()**2
                else:
                    backbone_norm += param_norm.item()**2
        head_norm = head_norm ** 0.5
        backbone_norm = backbone_norm ** 0.5
        return backbone_norm, head_norm

def extend_metrics(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        metrics = f(*args, **kwargs)
        time_metric = time.time()-start
        metrics.update({"time":time_metric})
        return metrics
    return wrapper

def get_attribute(obj, attr, *args):
    #Taken from: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/delta.py#L153
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class TwoHeadsModel(nn.Module):
    def __init__(self, model, source_head):
        super().__init__()
        self.model = model
        self.source_head = source_head

    def forward(self,*args, **kwargs):
        output, features = self.model(*args, **kwargs)
        x = self.model.last_layer_output
        source_output = None
        if isinstance(self.model._model, models.xcit.XCiT):
            if self.model._model.global_pool:
                x = x[:, 1:].mean(dim=1) if self.model._model.global_pool == 'avg' else x[:, 0]

        elif isinstance(self.model._model, models.swin_transformer_v2.SwinTransformerV2) or \
            isinstance(self.model._model, models.swin_transformer_v2_cr.SwinTransformerV2Cr) or \
            isinstance(self.model._model, models.swin_transformer.SwinTransformer):
            if self.model._model.global_pool == 'avg':
                x = x.mean(dim=1)

        elif isinstance(self.model._model,  models.volo.VOLO):
            if self.model._model.global_pool == 'avg':
                out = x.mean(dim=1)
            elif self.model._model.global_pool == 'token':
                out = x[:, 0]
            else:
                out = x
            out = self.source_head(out)
            if self.model._model.aux_head is not None:
                # generate classes in all feature tokens, see token labeling
                aux = self.model._model.aux_head(x[:, 1:])
                out = out + 0.5 * aux.max(1)[0]
            source_output = out

        if source_output is None:
            source_output = self.source_head(x)
        return output, source_output, features

class IntermediateLayerGetter(nn.Module):
    r"""
    Implementation from: https://github.com/thuml/Transfer-Learning-Library/blob/c4b0dfd1d0493f70958b715de5d7b12383954028/tllib/regularization/delta.py#L153

    Wraps a model to get intermediate output values of selected layers.
    Args:
       model (torch.nn.Module): The model to collect intermediate layer feature maps.
       return_layers (list): The names of selected modules to return the output.
       keep_output (bool): If True, `model_output` contains the final model's output, else return None. Default: True
    Returns:
       - An OrderedDict of intermediate outputs. The keys are selected layer names in `return_layers` and the values are the feature map outputs. The order is the same as `return_layers`.
       - The model's final output. If `keep_output` is False, return None.
    """
    def __init__(self, model, return_layers, pool_layer=None, keep_output=True):
        super().__init__()
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1,1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name in self.return_layers:
            layer = get_attribute(self._model, name)
            def hook(module, input, output, name=name):
                ret[name] = output
            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)

        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        # TODO: test if the gradients are computed thorugh the features for BSS
        # TODO: check if this works for all the considered networks
        self.last_layer_output = ret[self.return_layers[0]]
        features = self.pool_layer(self.last_layer_output.unsqueeze(-1))
        return output, features

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_layers(model, num_classes, device, freezable_thd=0, change_head=False):
    if isinstance(model, models.edgenext.EdgeNeXt):
        source_head = nn.Sequential(copy.deepcopy(model.head.global_pool),
                                    copy.deepcopy(model.head.norm),
                                    copy.deepcopy(model.head.flatten),
                                    copy.deepcopy(model.head.drop),
                                    copy.deepcopy(model.head.fc))
        if change_head:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
        last_hidden_layer_name = "norm_pre"
        head_name = "head"


    elif isinstance(model, models.dla.DLA):
        layers = list(model.children())
        source_head = nn.Sequential(copy.deepcopy(layers[-2]),
                                    copy.deepcopy(layers[-1]))

        if change_head:
            num_ftrs = layers[-2].in_channels
            model.fc = nn.Conv2d(num_ftrs, num_classes,
                                 layers[-2].kernel_size,
                                 layers[-2].stride, device=device)
        head = model.fc
        last_hidden_layer_name = "global_pool"
        head_name = "fc"


    elif isinstance(model, models.xcit.XCiT):
        source_head = copy.deepcopy(model.head)
        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)


        last_hidden_layer_name = "norm"
        head_name = "head"


    elif isinstance(model, models.byobnet.ByobNet):
        source_head = nn.Sequential(copy.deepcopy(model.head.global_pool),
                                    copy.deepcopy(model.head.fc),
                                    copy.deepcopy(model.head.flatten))
        if change_head:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)

        last_hidden_layer_name = "final_conv"
        head_name = "head"

    elif isinstance(model, models.cait.Cait):
        source_head = nn.Sequential(copy.deepcopy(model.head))

        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)

        last_hidden_layer_name = "norm"
        head_name = "head"

    elif isinstance(model, models.beit.Beit):
        source_head = nn.Sequential(copy.deepcopy(model.head))

        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)

        last_hidden_layer_name = "fc_norm"
        head_name = "head"

    elif isinstance(model, models.volo.VOLO):
        source_head = nn.Sequential(copy.deepcopy(model.head))

        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)

        last_hidden_layer_name = "norm"
        head_name = "head"

    elif isinstance(model, models.efficientnet.EfficientNet):
        source_head = nn.Sequential(copy.deepcopy(model.classifier))

        if change_head:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes).to(device)


        last_hidden_layer_name = "global_pool"
        head_name = "classifier"

    elif isinstance(model, models.convnext.ConvNeXt):
        source_head = nn.Sequential(copy.deepcopy(model.head.global_pool),
                                    copy.deepcopy(model.head.norm),
                                    copy.deepcopy(model.head.flatten),
                                    copy.deepcopy(model.head.drop),
                                    copy.deepcopy(model.head.fc))

        if change_head:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
        last_hidden_layer_name = "norm_pre"
        head_name = "head"

    elif isinstance(model, models.deit.VisionTransformer) or isinstance(model,
                                                                        models.vision_transformer.VisionTransformer):
        source_head = nn.Sequential(
            copy.deepcopy(model.head)
        )
        if change_head:
            num_ftrs = model.head.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)

        last_hidden_layer_name = "fc_norm"  # TODO @Sebastian: not sure...fc_norm is last layer but identity, "norm" would be last with an actual layer
        head_name = "head"

    elif isinstance(model, models.swin_transformer_v2.SwinTransformerV2) or \
            isinstance(model, models.swin_transformer_v2_cr.SwinTransformerV2Cr) or \
            isinstance(model, models.swin_transformer.SwinTransformer):
        source_head = nn.Sequential(
            copy.deepcopy(model.head)
        )
        if change_head:
            num_ftrs = model.head.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)

        last_hidden_layer_name = "norm"
        head_name = "head"

    else:
        raise NotImplementedError(f"model type {type(model)} not implemented for fine-tuning")

    hidden_layers = []
    head = []
    for name, module in model.named_modules():
        if not isinstance(module, type(model)):
            if name.startswith(head_name):
                head.append(module)
            else:
                hidden_layers.append(module)

    head = nn.ModuleList(head)
    counting_flags = []
    for layer in hidden_layers:
        count = count_parameters(layer)
        if count > freezable_thd:  # maybe thd should depend on the no. of params in the net.
            counting_flags.append(True)
        else:
            counting_flags.append(False)
    return last_hidden_layer_name, head_name, hidden_layers, head, source_head, counting_flags


def get_layers_by_module_children(model, num_classes, device, freezable_thd=0, change_head=False):

    if isinstance(model, models.edgenext.EdgeNeXt):
        source_head = nn.Sequential(copy.deepcopy(model.head.global_pool),
                                    copy.deepcopy(model.head.norm),
                                    copy.deepcopy(model.head.flatten),
                                    copy.deepcopy(model.head.drop),
                                    copy.deepcopy(model.head.fc))        
        if change_head:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
            
        stem, stages, norm_pre, head = list(model.children())
        hidden_layers = list(stem) + list(stages) + [norm_pre]
        last_hidden_layer_name = "norm_pre"
        head_name = "head"
  
    elif isinstance(model, models.dla.DLA):

        layers = list(model.children())
        hidden_layers = layers[:-2]       
        source_head = nn.Sequential(copy.deepcopy(layers[-2]),
                                    copy.deepcopy(layers[-1]))

        if change_head:
            num_ftrs = layers[-2].in_channels
            model.fc = nn.Conv2d(num_ftrs, num_classes, 
                                    layers[-2].kernel_size,
                                    layers[-2].stride, device=device)
        head = model.fc
        last_hidden_layer_name = "global_pool"
        head_name = "fc"


    elif isinstance(model, models.xcit.XCiT):
        source_head = copy.deepcopy(model.head)
        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)
            
        patch_embed, pos_embed, pos_drop, blocks,\
                cls_attn_blocks, norm, head = list(model.children())
        hidden_layers = [patch_embed, pos_embed, pos_drop] + list(blocks) +\
                        list(cls_attn_blocks) + [norm]
        last_hidden_layer_name = "norm"
        head_name = "head"


    elif isinstance(model, models.byobnet.ByobNet):
        source_head = nn.Sequential(copy.deepcopy(model.head.global_pool), 
                                    copy.deepcopy(model.head.fc),
                                    copy.deepcopy(model.head.flatten))
        if change_head:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
            
        stem, stages, final_conv, head = list(model.children())
        hidden_layers = [stem]+list(stages)+[final_conv]
        last_hidden_layer_name = "final_conv"
        head_name = "head.fc"
    
    elif isinstance(model, models.cait.Cait):
        source_head = nn.Sequential(copy.deepcopy(model.head))
        
        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)
    
        patch_embed, dropout, blocks, cls_attn_blocks, norm, head = list(model.children())
        hidden_layers = [patch_embed, dropout] + list(blocks) + list(cls_attn_blocks) + [norm]
        last_hidden_layer_name = "norm"
        head_name = "head"

    elif isinstance(model, models.beit.Beit):
        source_head = nn.Sequential(copy.deepcopy(model.head))
    
        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)
    
        patch_embed, dropout, blocks, identity, norm, head = list(model.children())
        hidden_layers = [patch_embed, dropout] + list(blocks) + [norm]
        last_hidden_layer_name = "fc_norm"
        head_name = "head"

    elif isinstance(model, models.volo.VOLO):
        source_head = nn.Sequential(copy.deepcopy(model.head))
    
        if change_head:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes).to(device)
    
        patch_embed, dropout, blocks, cls_attn_blocks, linear, norm, head = list(model.children())
        hidden_layers = [patch_embed, dropout] + list(blocks) + list(cls_attn_blocks) + [linear, norm]
        last_hidden_layer_name = "norm"
        head_name = "head"

    elif isinstance(model, models.efficientnet.EfficientNet):
        source_head = nn.Sequential(copy.deepcopy(model.classifier))
    
        if change_head:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes).to(device)
    
        conv2d_same, batch_norm, sequentials, conv2d, batch_norm2, adaptive_pool, head = list(model.children())
        hidden_layers = [conv2d_same, batch_norm] + list(sequentials) + [conv2d, batch_norm2, adaptive_pool]
        last_hidden_layer_name = "global_pool"
        head_name = "classifier"
    
    elif isinstance(model, models.convnext.ConvNeXt):
        source_head = nn.Sequential(copy.deepcopy(model.head.global_pool),
                                    copy.deepcopy(model.head.norm),
                                    copy.deepcopy(model.head.flatten),
                                    copy.deepcopy(model.head.drop),
                                    copy.deepcopy(model.head.fc))
    
        if change_head:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
    
        conv2d, convnextstage, identity, head = list(model.children())
        hidden_layers = list(conv2d) + list(convnextstage)
        last_hidden_layer_name = "norm_pre"
        head_name = "head"
    
    elif isinstance(model, models.deit.VisionTransformer) or isinstance(model, models.vision_transformer.VisionTransformer):
        source_head = nn.Sequential(
            copy.deepcopy(model.head)
            )
        
        if change_head:
            num_ftrs = model.head.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
        
        patch_embed, dropout, identity, blocks, layer_norm, identity, head = list(model.children())
        hidden_layers = [patch_embed, dropout] + list(blocks) + [layer_norm]
        last_hidden_layer_name = "fc_norm"  # TODO @Sebastian: not sure...fc_norm is last layer but identity, "norm" would be last with an actual layer
        head_name = "head"

    elif isinstance(model, models.swin_transformer_v2.SwinTransformerV2) or \
        isinstance(model, models.swin_transformer_v2_cr.SwinTransformerV2Cr) or \
        isinstance(model, models.swin_transformer.SwinTransformer):
        source_head = nn.Sequential(
            copy.deepcopy(model.head)
            )
    
        if change_head:
            num_ftrs = model.head.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes).to(device)
    
        patch_embed, dropout, blocks, layer_norm, head = list(model.children())
        hidden_layers = [patch_embed, dropout] + list(blocks) + [layer_norm]
        last_hidden_layer_name = "norm"
        head_name = "head"

    else:
        raise NotImplementedError(f"model type {type(model)} not implemented for fine-tuning")

    counting_flags = []
    for layer in hidden_layers:
        count = count_parameters(layer)
        if count > freezable_thd: #maybe thd should depend on the no. of params in the net.
            counting_flags.append(True)
        else:
            counting_flags.append(False)
    return last_hidden_layer_name, head_name, hidden_layers, head, source_head, counting_flags


def prepare_model_for_finetuning(
        model,
        num_classes: int = 1000,
        pct_to_freeze: float = 1.0,
        freezable_thd: float = 0.,
        return_features: bool = False,
        return_source_output: bool = False,
        change_head: bool = True,
        device: str = "cuda"):

    if isinstance(model, TwoHeadsModel):
        temp_source_head = model.source_head
        model = model.model
    else:
        temp_source_head = None

    if isinstance(model, IntermediateLayerGetter):
        model = model._model

    #Adds head and freeze the layers
    last_hidden_layer_name, head_name, hidden_layers, \
        head, source_head, countable_flags = get_layers(model, num_classes, device,
                                                                freezable_thd=freezable_thd,
                                                                change_head=change_head)
    if temp_source_head:
        source_head = temp_source_head

    n_countable_layers = sum(countable_flags)

    for param in model.parameters():
        param.requires_grad = False

    for param in head.parameters():
        param.requires_grad = True

    pos = 0
    for layer, is_countable in zip(hidden_layers, countable_flags):
        if is_countable: pos += 1
        if pos / n_countable_layers > pct_to_freeze:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False
    if return_features:
        model = IntermediateLayerGetter(model, [last_hidden_layer_name])
        head_name = "_model." + head_name
        if return_source_output:
            model = TwoHeadsModel(model, source_head)
            head_name = "model."+head_name
    return model, head_name


def get_dataset_name(tfds_dataset_name):
    """ tfds/cycle_gan/vangogh2photo --> cycle_gan/vangogh2photo"""
    return tfds_dataset_name.split("/", 1)[1]


def get_dataset_path(dataset_dir, tfds_dataset_name):
    """ inputs: dataset_dir, tfds_dataset_name (tfds/cycle_gan/vangogh2photo) --> {dataset_dir}/cycle_gan/vangogh2photo"""
    dataset_name = get_dataset_name(tfds_dataset_name)
    return os.path.join(os.getcwd(), dataset_dir, dataset_name)


def get_icgen_dataset_info_json(dataset_path, dataset_name):
    dataset_name_wo_tfds = get_dataset_name(dataset_name)
    dataset_name = dataset_name_wo_tfds.replace('/', '_')
    # dataset_name = dataset_path.split("/")[-1]
    dataset_info_json_path = os.path.join(dataset_path, f"icgen_info_{dataset_name}.json")
    with open(dataset_info_json_path, 'r') as f:
        return json.load(f)
    return None


def get_number_of_classes(dataset_path):
    """ reads the label.labels.txt embedded somewhere in the dataset path and parses the file in to a list"""
    file_path = find_file(dataset_path, "features.json")
    
    if file_path:
        with open(file_path, 'r') as f:
            features = json.load(f)
            return int(features["featuresDict"]["features"]["label"]["classLabel"]["numClasses"])
    else:
        file_path = find_file(dataset_path, "label.labels.txt")
        with open(file_path, 'r') as f:
            label_list = [line.split("\n")[0] for line in f]
            return len(label_list)
    
