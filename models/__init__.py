import torch
import torch.nn as nn
from torchvision import models
import timm
# 當創建新環境時，需要將ultralytics/nn/modules內的Classify class做更動
from ultralytics.nn.tasks import ClassificationModel


def get_optim(optim_name:str, model, lr:float):
    """optim_name = [adam | adamw | sgd | rmsprop | adagrad"""
    optim_name = optim_name.lower()
    if optim_name == "adam":
        return torch.optim.Adam(model.parameters(),lr = lr)
    elif optim_name == "adamw":
        return torch.optim.AdamW(model.parameters(),lr = lr)
    elif optim_name == "sgd":
        return torch.optim.SGD(model.parameters(),lr = lr)
    elif optim_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(),lr = lr)
    elif optim_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(),lr = lr)
    else:
        print(f'Don\'t find the model: {optim_name} . default optimizer is adam')
        return torch.optim.Adam(model.parameters(),lr = lr)
    
def initialize_model(model_name, num_classes=None, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = timm.create_model('resnet50',pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = timm.create_model('resnet152',pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "yolov8":
        """Yolov8 by ultralytic, please install the mudule ultralytic first"""
        if use_pretrained:
            model_obj = ClassificationModel(cfg='yolov8n-cls.yaml', # build a new model from YAML
                                        model=None,ch=3,
                                        nc= num_classes,
                                        cutoff=10,verbose=True)
            model_ft = model_obj.model.cpu()
            model_ft.load_state_dict(torch.load("yolov8n-cls.pt",map_location='cpu'),strict=False)
        else:
            model_obj = ClassificationModel(cfg='yolov8n-cls.yaml', # build a new model from YAML
                                        model=None,ch=3,
                                        nc= num_classes,
                                        cutoff=10,verbose=True)  
            model_ft = model_obj.model
    elif model_name == "encoder":
        """ Resnet50
        """
        model_ft = timm.create_model('resnet152',pretrained=use_pretrained)

        model_ft.layer2 = nn.Identity()
        model_ft.layer3 = nn.Identity()
        model_ft.layer4 = nn.Identity()
        model_ft.fc = nn.Identity()
        model_ft.global_pool = nn.Identity()

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft