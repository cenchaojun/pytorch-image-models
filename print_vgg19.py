import torchvision
import torch.nn as nn
model_conv = torchvision.models.vgg19(pretrained=True)
print(model_conv.classifier._modules)
model_conv.classifier._modules.pop('4')
model_conv.classifier._modules.pop('5')
model_conv.classifier._modules.pop('6')
print(model_conv.classifier._modules)
# print(model_conv)
# crop = nn.Sequential(*list(model_conv.classifier._modules.children())[:-1])
# print(crop)
# model_conv.classifier._modules['6'] = None

# print(model_conv.classifier._modules)
# print(type(model_conv))