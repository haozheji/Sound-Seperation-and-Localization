# visual_model.py
import torch
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class my_resnet_model(ResNet):
    def __init__(self, pram_k=16, modelUrl=model_urls['resnet18']):
        super(my_resnet_model, self).__init__(BasicBlock, [2, 2, 2, 2], 1000)
        self.pram_k = pram_k
        self.load_state_dict(model_zoo.load_url(modelUrl))
        del self.fc
        del self.avgpool
        self.conv3x3 = nn.Conv2d(512, self.pram_k, kernel_size=3)
        self.last_maxpool = nn.MaxPool2d(kernel_size=5)



    def forward(self, x):   # input bs * 3 * 224 * 224
        bs = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # bs * 512 * 7 * 7

        x = self.conv3x3(x) # bs * K * 7 * 7

        x = self.last_maxpool(x).squeeze()
        return x
        # x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3))
        # x = x.view(x.size(0), -1)
        # x = self.myfc(x)    # bs * K , 2
        # x = x.view(bs, self.pram_k, x.size(1))

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x    # final output bs * K * 2

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)    # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 1024
        x = self.up1(x5, x4) # 512
        x = self.up2(x, x3)  # 256
        x = self.up3(x, x2)  #
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

# class UNet(nn.Module):

#     def __init__(self, n_channels, n_class):
#         super(UNet, self).__init__()

#         # Use ResNet18 as the encoder with the pretrained weights
#         self.base_model = models.resnet18(pretrained=True)
#         self.base_layers = list(self.base_model.children())
#         # self.conv1x1 = nn.Conv2d(1,3,1)

#         self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
#         self.layer1_1x1 = convrelu(256, 256, 1, 0)
#         self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
#         self.layer2_1x1 = convrelu(512, 512, 1, 0)
#         self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
#         self.layer3_1x1 = convrelu(1024, 512, 1, 0)
#         self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
#         self.layer4_1x1 = convrelu(2048, 1024, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
#         self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
#         self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
#         self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

#         self.conv_original_size0 = convrelu(n_channels, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

#         self.conv_last = nn.Conv2d(64, n_class, 1)

#     def forward(self, input):
#         x_original = self.conv_original_size0(input)
#         x_original = self.conv_original_size1(x_original)

#         layer0 = self.layer0(input)
#         layer1 = self.layer1(layer0)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)

#         # Upsample the last/bottom layer
#         layer4 = self.layer4_1x1(layer4)
#         x = self.upsample(layer4)
#         # Create the shortcut from the encoder
#         layer3 = self.layer3_1x1(layer3)
#         x = torch.cat([x, layer3], dim=1)
#         x = self.conv_up3(x)

#         x = self.upsample(x)
#         layer2 = self.layer2_1x1(layer2)
#         x = torch.cat([x, layer2], dim=1)
#         x = self.conv_up2(x)

#         x = self.upsample(x)
#         layer1 = self.layer1_1x1(layer1)
#         x = torch.cat([x, layer1], dim=1)
#         x = self.conv_up1(x)

#         x = self.upsample(x)
#         layer0 = self.layer0_1x1(layer0)
#         x = torch.cat([x, layer0], dim=1)
#         x = self.conv_up0(x)

#         x = self.upsample(x)
#         x = torch.cat([x, x_original], dim=1)
#         x = self.conv_original_size2(x)

#         out = self.conv_last(x)

#         return out

class VisualAudioModel(nn.Module):

    '''
    the warpped model
    input: audio stft map and visual input image
    output: bs * 2 * 256 * 256 with two localized sounds stft maps
    '''

    def __init__(self, pram_k_ = 16, use_model=model_urls['resnet18']):
        super(VisualAudioModel, self).__init__()
        self.resnet = my_resnet_model(pram_k=pram_k_, modelUrl=use_model)
        #self.left_resnet = my_resnet_model(pram_k=pram_k_, modelUrl=use_model)
        self.unet_layer = UNet(1, pram_k_)
        self.pram_k = pram_k_
        #self.out_left = nn.Linear(pram_k_, 1)
        self.out = nn.Linear(pram_k_, 1)

    def forward(self, a_input, v_input):
        if(len(a_input.shape)==3):
            a_input = a_input.unsqueeze(1)
        #print(a_input.size())
        a_fea  = self.unet_layer(a_input)     # output bs * K * 256 * 256
        ori_size = v_input.size()
        input_size = (ori_size[0] * ori_size[1], ori_size[2], ori_size[3], ori_size[4])
        v_input = v_input.view(input_size)
        #lv_input = lv_input.view(input_size)

        v_fea = self.resnet(v_input) # output bs*T, K
        #lv_fea = self.resnet(lv_input) # output bs*T, K
        v_fea = v_fea.view(ori_size[0], ori_size[1], -1)
        #lv_fea = lv_fea.view(ori_size[0], ori_size[1], -1)
        v_fea = torch.max(v_fea, dim=1)[0]
        #lv_fea = torch.max(lv_fea, dim=1)[0]# size: bs x K
        # the size of the 1x1 kernel is bs * K * 256 * 256
        prod = (a_fea * v_fea.unsqueeze(2).unsqueeze(3).expand_as(a_fea)).view(v_fea.size(0), -1, v_fea.size(1))
        Out = self.out(prod).squeeze(2)
        Out = F.sigmoid(Out.view(v_fea.size(0), a_fea.size(2), -1))

        #l_prod = (a_fea * lv_fea.unsqueeze(2).unsqueeze(3).expand_as(a_fea)).view(lv_fea.size(0), -1, lv_fea.size(1))
        #l_out = self.out(l_prod).squeeze(2)
        #l_out = F.sigmoid(l_out.view(lv_fea.size(0), a_fea.size(2), -1))
        #print(l_out)
        return Out# both with bs * 256 * 256


    def _forward(self, a_input, rv_input, lv_input):
        # the given input audio is bs * 1 * 256 * 256
        # the given input visual is bs * temporl * 3 * 224 * 224
        # lv_input.transpose_(2, 4)
        # rv_input.transpose_(2, 4)
        if(len(a_input.shape)==3):
            a_input = a_input.unsqueeze(1)
        a_fea  = self.unet_layer(a_input)     # output bs * K * 256 * 256
        ori_size = rv_input.size()
        input_size = (ori_size[0] * ori_size[1], ori_size[2], ori_size[3], ori_size[4])
        rv_input = rv_input.view(input_size)
        lv_input = lv_input.view(input_size)

        rv_fea = self.resnet(rv_input) # output bs*T, K
        lv_fea = self.resnet(lv_input) # output bs*T, K
        rv_fea = rv_fea.view(ori_size[0], ori_size[1], -1)
        lv_fea = lv_fea.view(ori_size[0], ori_size[1], -1)
        rv_fea = torch.max(rv_fea, dim=1)[0]
        lv_fea = torch.max(lv_fea, dim=1)[0]# size: bs x K
        # the size of the 1x1 kernel is bs * K * 256 * 256
        r_prod = (a_fea * rv_fea.unsqueeze(2).unsqueeze(3).expand_as(a_fea)).view(rv_fea.size(0), -1, rv_fea.size(1))
        r_out = self.out(r_prod).squeeze(2)
        r_out = F.sigmoid(r_out.view(rv_fea.size(0), a_fea.size(2), -1))

        l_prod = (a_fea * lv_fea.unsqueeze(2).unsqueeze(3).expand_as(a_fea)).view(lv_fea.size(0), -1, lv_fea.size(1))
        l_out = self.out(l_prod).squeeze(2)
        l_out = F.sigmoid(l_out.view(lv_fea.size(0), a_fea.size(2), -1))
        #print(l_out)
        return r_out, l_out# both with bs * 256 * 256




if __name__ =='__main__':

    # global parameters
    use_cuda = True
    bs = 2
    # end global parm

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model  = VisualAudioModel().to(device)
    input_a = torch.randn(bs, 256,256).to(device)
    input_v = torch.randn(bs, 3, 224,224).to(device)
    print "input shape is:",input_v.size(),input_a.size()
    out_r, out_l  = model(input_a, input_v, input_v)
    print out_r.size(),out_l.size()




        #v_fea_left = v_fea[:,:,0].unsqueeze(2).unsqueeze(3).repeat(1,1,256,256)
        #v_fea_right = v_fea[:,:,1].unsqueeze(2).unsqueeze(3).repeat(1,1,256,256)
        # the output is bs * 1 * 256 * 256
        #left_out = torch.sum(v_fea_left * a_fea, dim = 1, keepdim = True)
        #right_out = torch.sum(v_fea_right * a_fea, dim = 1, keepdim = True)
        #out = torch.cat((left_out,right_out),1)
        #return out



# class VAN(ResNet):
#     def __init__(self, pram_k=16, modelUrl=model_urls['resnet18']):
#         super(VAN, self).__init__(BasicBlock, [2, 2, 2, 2], 1000)
#         self.pram_k = pram_k
#         self.load_state_dict(model_zoo.load_url(modelUrl))
#         del self.fc
#         del self.avgpool
#         del self.layer4



#     def forward(self, x):   # input bs * 3 * 224 * 224
#         bs = x.size(0)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)


#         return x    # final output bs * K * 2






# class localize_model(nn.Module):
#     def __init__(self, pram_k = 16):
#         super(localize_model, self).__init__()
#         self.conv_loc = my_resnet_model(pram_k)
#         self.fc_loc = nn.Linear(pram_k * 2, 2)

#     def forward(self, x):
#         x = self.conv(x)    # bs * K * 2
#         x = x.size(x.size(0),-1)
#         x = self.fc_loc(F.relu(x))  # bs * 2

# class segment_model(nn.Module):
#     def __init__(self, pram_k = 16):
#         super(segment_model, self).__init__()
#         self.conv_seg = UNet(1, pram_k)


    # def forward(self, x):
    #     x = UNet(x)    # bs * k * 256 * 256






