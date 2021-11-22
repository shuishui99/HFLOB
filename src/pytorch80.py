import torch
from torch import nn
from torch.nn import functional as F

class ResBlok(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        :param stride:
        """
        super(ResBlok, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        if ch_in != ch_out:
            # [b, ch_in, h, w] -> [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):
        '''

        :param x:[b, ch, h, w]
        :return:
        '''

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        # print("con shape", out.shape)

        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        '''
        input shape: [b, 3, 32, 32]
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

#         [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlok(64, 128, stride=2)
#         [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlok(128, 256, stride=2)
        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlok(256, 512, stride=2)
        # [b, 512, h, w] => [b, 512, h, w]
        self.blk4 = ResBlok(512, 512, stride=2)

        self.outlayer = nn.Linear(512*1*1, 10)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''

        x = F.relu(self.conv1(x))
        print("before blk: ", x.shape)

#         [b, 64, h, w] => [b, 512, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        print("after blk: ", x.shape)
        # [b, 512, h ,w ] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        print("after pool: ", x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x



def main():
    # blk = ResBlok(64, 128)
    # tmp = torch.randn(2, 64, 32, 32)
    # out = blk(tmp)
    # print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print("resnet:", out.shape)

if __name__ == '__main__':
    main()