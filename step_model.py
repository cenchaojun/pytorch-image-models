import timm
import torch

model_list = timm.list_models()
print(model_list)
x = torch.randn([1,3,224,224])
model_resnet50 = timm.create_model('resnet50',pretrained=False)
print(model_resnet50)
out = model_resnet50(x)
print(out.shape)