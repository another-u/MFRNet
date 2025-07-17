from MFRNet import MFRNet
import torch

from thop import profile


if __name__ == '__main__':
    model = MFRNet().cuda()
    model.load_state_dict(torch.load('./MFRNet_Baseline_3_Scale_train_pth/Net_epoch_best.pth'))
    total = sum([param.nelement() for param in model.parameters()])
    print('params:%.2fM' % (total / 1000000))

    input = torch.randn(1, 3, 384, 384)
    input = input.cuda()
    Flops, params = profile(model, inputs=(input,), verbose=False)

    print('Flops:%.4fG'%(Flops/1000000000))
    print('params:%.4fM'%(params/1000000))