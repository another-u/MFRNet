import torch
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
import random
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class ReductionLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ReductionLayer, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Conv2d(32*32, 32, 7, 1, 3)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        m_batchsize1, C1, height1, width1 = avg_out.size()
        m_batchsize2, C2, height2, width2 = max_out.size()
        avg_out = avg_out.view(m_batchsize1, C1, -1)
        max_out = max_out.view(m_batchsize2, C2, -1).permute(0, 2, 1)
        energy = torch.bmm(avg_out,max_out)
        attention = self.softmax(energy)
        attention = attention.view(m_batchsize1, 32*32, height1, width1)
        out = self.conv(attention)

        # out = avg_out + max_out
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(28 * 28, 1, 3, 1, 1)  # can be annotated
        self.conv_4 = nn.Conv2d(96 * 96, 1, 3, 1, 1) # can be annotated

        self.conv_1 = nn.Conv2d(48 * 48, 1, 3, 1, 1)
        self.conv_2 = nn.Conv2d(24 * 24, 1, 3, 1, 1)
        self.conv_3 = nn.Conv2d(12 * 12, 1, 3, 1, 1)

        self.conv1 = nn.Conv2d(33, 1, kernel_size, padding=padding, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x_shape:{}".format(x.shape))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        batchsize1, C1, H1, W1 = avg_out.size()
        batchsize2, C2, H2, W2 = max_out.size()

        avg_out = avg_out.view(batchsize1, -1, H1*W1).permute(0,2,1)
        max_out = max_out.view(batchsize2, -1, H2*W2)
        x_mid = torch.bmm(avg_out, max_out)
        attention = self.softmax(x_mid)
        attention = attention.view(batchsize1, H1*W1, H1, W1)
        if H1==48 and W1==48:
            attention = self.conv_1(attention)
        elif H1==24 and W1==24:
            attention = self.conv_2(attention)
        elif H1==12 and W1==12:
            attention = self.conv_3(attention)

        out = torch.cat([attention, x], dim=1)
        out = self.conv1(out)
        return out


class BidirectionalFeatureTransferModule(nn.Module):
    def __init__(self, dim):
        super(BidirectionalFeatureTransferModule, self).__init__()

        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()

    def forward(self, x, g):
        x_ = x.clone()
        g_ = g.clone()

        x_0 = self.ca(x_)
        g_0 = self.sa(g_)

        ## Bidirectional Interaction
        # channel attention
        x_sig = torch.sigmoid(x_0)
        g_att_2 = x_sig * g_
        # spatial attention
        g_sig = torch.sigmoid(g_0)
        x_att_2 = g_sig * x_

        out_1 = g_att_2
        out_2 = x_att_2

        return out_1, out_2

class single_to_multi_Guidance(nn.Module):
    def __init__(self, channel):
        super(single_to_multi_Guidance, self).__init__()

        self.conv = nn.Conv2d(32, 1, 1, 1)
        self.conv_1 = nn.Conv2d(33, 32, 3, 1, 1)

    def forward(self, x, y):
        y_ = self.conv(y)
        # reverse guided block
        y_ = -1 * (torch.sigmoid(y_)) + 1
        x_cat = torch.cat((y_, x), dim=1)

        out = self.conv_1(x_cat) + x
        return out

class multi_to_single_Guidance(nn.Module):
    def __init__(self, channel):
        super(multi_to_single_Guidance, self).__init__()

        self.conv = nn.Conv2d(32, 1, 1, 1)
        self.conv_1 = nn.Conv2d(33, 32, 3, 1, 1)

    def forward(self, x, y):
        B, C, H, W = y.size()
        x = x.expand(B, -1, -1, -1)
        y_ = self.conv(y)
        # reverse guided block
        y_ = -1 * (torch.sigmoid(y_)) + 1
        x_cat = torch.cat((y_, x), dim=1)

        out = self.conv_1(x_cat) + x
        return out

class MutualGuidanceFusionModule(nn.Module):
    def __init__(self, channel):
        super(MutualGuidanceFusionModule, self).__init__()

        self.m_to_s = multi_to_single_Guidance(channel)
        self.bftm = BidirectionalFeatureTransferModule(32)

    def forward(self, x, y):
        x_ = self.m_to_s(x, y)
        x_0, y_0 = self.bftm(x_, y)

        return x_0, y_0


class selfAttention(nn.Module):
    def __init__(self, channel):
        super(selfAttention, self).__init__()
        self.gat1 = GATConv(channel, channel, heads=4, concat=True)
        self.gat2 = GATConv(4 * channel, channel, heads=1, concat=False)

        self.softmax = nn.Softmax(dim=-1)

        self.conv_1 = nn.Conv2d(48 * 48, channel, 3, 1, 1)
        self.conv_2 = nn.Conv2d(24 * 24, channel, 3, 1, 1)
        self.conv_3 = nn.Conv2d(12 * 12, channel, 3, 1, 1)

        self.conv = nn.Conv2d(2 * channel, channel, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()

    def image_to_graph(self, image_tensor):
        """
        Transform image Tensor into graph data (node features and edge information)
        """
        batch_size, channels, height, width = image_tensor.size()

        # Reshaping images into node features: (batch_size * height * width, channels)
        x = image_tensor.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, num_nodes, channels)
        x = x.flatten(0, 1)  # (batch_size * num_nodes, channels)

        # Creating an edge index: using nearest neighbor joins to construct edges
        edge_index = self.create_edge_index(height, width, batch_size)

        # Creating graph data
        graph_data = Data(x=x, edge_index=edge_index)
        return graph_data

    def create_edge_index(self, height, width, batch_size):
        """
        Create edge indexes to connect nearest neighbor pixels (e.g., 8-neighborhood)
        """
        edges = []
        for row in range(height):
            for col in range(width):
                node_idx = row * width + col
                neighbors = [
                    (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),  # Up and down, left and right neighbors
                    (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)  # diagonal neighbor
                ]

                #For each neighbor, check if it is within the boundaries of the image
                for (nbr_row, nbr_col) in neighbors:
                    if 0 <= nbr_row < height and 0 <= nbr_col < width:
                        nbr_idx = nbr_row * width + nbr_col
                        edges.append((node_idx, nbr_idx))

        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_index = edge_index.cuda()
        return edge_index

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x_graph = self.image_to_graph(x)
        x_k = self.gat2(F.relu(self.gat1(x_graph.x, x_graph.edge_index)), x_graph.edge_index)
        x_q = self.gat2(F.relu(self.gat1(x_graph.x, x_graph.edge_index)), x_graph.edge_index)
        x_v = self.gat2(F.relu(self.gat1(x_graph.x, x_graph.edge_index)), x_graph.edge_index)

        x_k = x_k.view(batch_size, -1, height * width).permute(0, 2, 1)
        x_q = x_q.view(batch_size, -1, height * width)
        x_mid = torch.bmm(x_k, x_q)
        x_att = self.softmax(x_mid)
        x_att = x_att.view(batch_size, height * width, height, width)
        if height == 48 and width == 48:
            x_att = self.conv_1(x_att)
        elif height == 24 and width == 24:
            x_att = self.conv_2(x_att)
        elif height == 12 and width == 12:
            x_att = self.conv_3(x_att)

        x_v = x_v.view(batch_size, channels, height, width)
        x_out = torch.cat([x_att, x_v], dim=1)
        x_out = self.conv(x_out)

        x_out = self.gamma * x_out + x
        x_out_graph = self.image_to_graph(x_out)
        x_out = self.gat2(F.relu(self.gat1(x_out_graph.x, x_out_graph.edge_index)), x_out_graph.edge_index)
        x_out = x_out.view(batch_size, channels, height, width)

        x_out = self.sigmoid(x_out)
        return x_out

class SelfAttGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(SelfAttGraphConv, self).__init__()
        self.selfAtt_1 = selfAttention(32)
        self.selfAtt_2 = selfAttention(32)
        self.selfAtt_3 = selfAttention(32)

    def forward(self, x1, x2, x3):
        """
        x1, x2, x3: -------------------------------shape:
        [3, 32, 48, 48], [3, 32, 24, 24], [3, 32, 12, 12]
        """
        # self-attention
        x1_out = self.selfAtt_1(x1)
        x2_out = self.selfAtt_2(x2)
        x3_out = self.selfAtt_3(x3)

        return x1_out, x2_out, x3_out


class DualGlobalPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DualGlobalPooling, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)
        self.conv_res_o = nn.Conv2d(in_channel, out_channel, 1)
        self.sigmoid_avg = nn.Sigmoid()
        self.sigmoid_max = nn.Sigmoid()

        self.gamma = nn.Parameter(torch.ones(1))
        self.conv_cat_1 = BasicConv2d(2 * out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        x_cat_0 = self.globalAvgPool(x)
        x_cat_0 = self.conv_res(x_cat_0)
        x_cat_att0 = self.sigmoid_avg(x_cat_0)

        x_cat_1 = self.globalMaxPool(x)
        x_cat_1 = self.conv_res_o(x_cat_1)
        x_cat_att1 = self.sigmoid_max(x_cat_1)

        x_cat_0 = self.gamma * x_cat_att0.expand_as(x) + x
        x_cat_1 = self.gamma * x_cat_att1.expand_as(x) + x

        out = self.conv_cat_1(torch.cat((x_cat_0, x_cat_1), 1))
        return out

class SpatialContextExplorationModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpatialContextExplorationModule, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            DualGlobalPooling(out_channel, out_channel)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            DualGlobalPooling(out_channel, out_channel)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            DualGlobalPooling(out_channel, out_channel)
        )
        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = self.conv_cat(torch.cat((x0, x1, x2), 1))
        return out


class MFRNet(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(MFRNet, self).__init__()
        # ---- Res2Net Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

        # ------Channel reduction-----
        self.reduce_conv4 = ReductionLayer(512, channel)
        self.reduce_conv5 = ReductionLayer(1024, channel)
        self.reduce_conv6 = ReductionLayer(2048, channel)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.selfAttGAT = SelfAttGraphConv(32,32,heads=4)

        self.scem_1 = SpatialContextExplorationModule(32, 32)
        self.scem_2 = SpatialContextExplorationModule(32, 32)
        self.scem_3 = SpatialContextExplorationModule(32, 32)

        self.mgfm = MutualGuidanceFusionModule(channel)
        self.s_to_m = single_to_multi_Guidance(channel)

        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x):
        # --------------Single-------------------
        a = len(x)
        b = random.randint(0, a - 1)
        x_1 = x[b]
        C, H, W = x_1.size()
        x_s = x_1.view(-1, C, H, W)

        x_s = self.resnet.conv1(x_s)
        x_s = self.resnet.bn1(x_s)
        x_s = self.resnet.relu(x_s)
        x_s = self.resnet.maxpool(x_s)  # bs, 64, 64, 64
        x_s1 = self.resnet.layer1(x_s)  # bs, 256, 64, 64
        x_s2 = self.resnet.layer2(x_s1)  # bs, 512, 32, 32
        x_s3 = self.resnet.layer3(x_s2)  # 16,16
        x_s4 = self.resnet.layer4(x_s3)  # bs, 2048, 8, 8

        # Channel reduction
        cr2 = self.reduce_conv4(x_s2)
        cr3 = self.reduce_conv5(x_s3)
        cr4 = self.reduce_conv6(x_s4)

        cr2, cr3, cr4 = self.selfAttGAT(cr2, cr3, cr4)

        # --------------Couple-------------------
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 64, 64
        x1 = self.resnet.layer1(x)  # bs, 256, 64, 64
        x2 = self.resnet.layer2(x1)  # bs, 512, 32, 32
        x_c3 = self.resnet.layer3(x2)  # 16,16
        x_c4 = self.resnet.layer4(x_c3)  # bs, 2048, 8, 8

        x4_rfb = self.reduce_conv4(x2)  # channel---->32 48
        x5_rfb = self.reduce_conv5(x_c3)  # channel---->32 24
        x6_rfb = self.reduce_conv6(x_c4)  # channel---->32 12

        x4_rfb = self.scem_1(x4_rfb)
        x5_rfb = self.scem_2(x5_rfb)
        x6_rfb = self.scem_3(x6_rfb)

        # mutual guidance (include single_to_multi_Guidance)
        m1_1, m1_2 = self.mgfm(cr2, x4_rfb)
        m2_1, m2_2 = self.mgfm(cr3, x5_rfb)
        m3_1, m3_2 = self.mgfm(cr4, x6_rfb)
        # single_to_multi_Guidance
        col_1 = self.s_to_m(m3_1, m3_2)
        col_2 = self.s_to_m(self.upsample2(col_1) + m2_1, m2_2)
        col_3 = self.s_to_m(self.upsample2(col_2) + m1_1, m1_2)

        col_3 = self.conv3(col_3)
        # -------------------heat map---------------------------------
        pred = F.interpolate(col_3, scale_factor=8, mode='bilinear')

        return pred


if __name__ == '__main__':
    import os
    import torch.backends.cudnn as cudnn

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True
    net = MFRNet(imagenet_pretrained=False).cuda()
    net.eval()

    dump_x = torch.randn(5, 3, 384, 384)
    dump_x = dump_x.cuda()
    out = net(dump_x)
    print(out.shape)
