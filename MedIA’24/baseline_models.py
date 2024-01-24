from Models.generate_model import *
from Models.res2net import res2net50_v1b_26w_4s,res2net50_v1b_14w_8s,res2net101_v1b_26w_4s
import torch.nn.functional as F
class Medical_feature_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_feature_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 128, 128
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 128, 128
        x2 = self.res2net.layer2(x1)     # bs, 512, 64, 64
        x3 = self.res2net.layer3(x2)     # bs, 1024, 32, 32
        x4 = self.res2net.layer4(x3)     # bs, 2048, 16, 16
        return x4


class Medical_base_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_base_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 128, 128
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 128, 128
        x2 = self.res2net.layer2(x1)     # bs, 512, 64, 64
        x3 = self.res2net.layer3(x2)     # bs, 1024, 32, 32
        x4 = self.res2net.layer4(x3)     # bs, 2048, 16, 16
        x4 = self.res2net.avgpool(x4)    # bs, 2048, 1, 1
        x4 = x4.view(x4.size(0), -1)  # bs, 1， 2048,
        return x4

class Medical_base2_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_base2_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_14w_8s(pretrained=True)
        # self.res2net = res2net50_v1b_26w_4s(pretrained=True)
        # self.res2net = res2net101_v1b_26w_4s(pretrained=True)

    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 64, 64
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 64, 64
        x2 = self.res2net.layer2(x1)     # bs, 512, 32, 32
        x3 = self.res2net.layer3(x2)     # bs, 1024, 16, 16
        x4 = self.res2net.layer4(x3)     # bs, 2048, 8, 8
        x4 = self.res2net.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        return x4

class Medical_base_dropout_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_base_dropout_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 64, 64
        # dropout layer
        x = F.dropout(x, p=0.2)
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 64, 64
        x2 = self.res2net.layer2(x1)     # bs, 512, 32, 32
        x3 = self.res2net.layer3(x2)     # bs, 1024, 16, 16
        x4 = self.res2net.layer4(x3)     # bs, 2048, 8, 8
        # dropout layer
        x4 = F.dropout(x4, p=0.2)
        x4 = self.res2net.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        return x4

class Medical_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 64, 64
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 64, 64
        x2 = self.res2net.layer2(x1)     # bs, 512, 32, 32
        x3 = self.res2net.layer3(x2)     # bs, 1024, 16, 16
        x4 = self.res2net.layer4(x3)     # bs, 2048, 8, 8
        x4 = self.res2net.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        out = self.fc(x4)
        return out


class Medical_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)
        if classifier_OCT_dims[0][0] == 128:
            self.fc = nn.Linear(8192, num_classes) # MMOCTF
        else:
            self.fc = nn.Linear(3072, num_classes) # OLIVES

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 64, 64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 256, 64, 64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 512, 32, 32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 1024, 16, 16
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 2048, 8, 8
        x4 = self.resnet_3DNet.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        out = self.fc(x4)
        return out

class Medical_base_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_base_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 32, 32,64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 64, 32, 32,64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 128, 16, 16,32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 256, 16, 16,32
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 512, 16, 16，32
        x4 = self.resnet_3DNet.avgpool(x4) # bs, 512, 16, 1，1
        x4 = x4.view(x4.size(0), -1) # 8192
        return x4

class Medical_feature_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_feature_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 32, 32,64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 64, 32, 32,64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 128, 16, 16,32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 256, 16, 16,32
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 512, 16, 16，32
        return x4

class Medical_base2_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_base2_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=18, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='A',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_18_23dataset.pth', nb_class=num_classes)

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 32, 32,64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  ## bs, 64, 32, 32,64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 128, 16, 16,32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 256, 16, 16,32
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 512, 16, 16，32
        x4 = self.resnet_3DNet.avgpool(x4)  # bs, 512, 16, 1，1
        x4 = x4.view(x4.size(0), -1) # 8192
        return x4

class Medical_base_dropout_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_base_dropout_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 64, 64
        # dropout layer
        x = F.dropout(x, p=0.2)
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 256, 64, 64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 512, 32, 32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 1024, 16, 16
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 2048, 8, 8
        x4 = self.resnet_3DNet.avgpool(x4)
        # dropout layer
        x4 = F.dropout(x4, p=0.2)
        x4 = x4.view(x4.size(0), -1)
        return x4

class ResNet3D(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(ResNet3D, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims
        self.resnet_3DNet = Medical_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.resnet_3DNet])
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sfm = nn.Softmax()

    def forward(self, X, y):
        output = self.infer(X[1])
        loss = 0
        for v_num in range(self.modalties):
            pred = output[v_num]
            # label = F.one_hot(y, num_classes=self.classes)
            # loss = self.ce_loss(label, pred)
            loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input)
            evidence[m_num] = self.sfm(backbone_output)
        return evidence

class Res2Net2D(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Res2Net2D, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        classifier_Fundus_dims = classifiers_dims[0]
        self.res2net_2DNet = Medical_2DNet(num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet])
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sfm = nn.Softmax()

    def forward(self, X, y):
        output = self.infer(X[0])
        loss = 0
        for v_num in range(self.modalties):
            pred = output[v_num]
            # label = F.one_hot(y, num_classes=self.classes)
            # loss = self.ce_loss(label, pred)
            loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input)
            evidence[m_num] = self.sfm(backbone_output)
        return evidence

class Multi_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        if classifier_OCT_dims[0][0] == 128:
            self.fc = nn.Linear(2048 + 8192, classes) # MMOCTF
        else:
            self.fc = nn.Linear(2048 + 3072, classes) # OLIVES

        # self.fc = nn.Linear(2048 + 8192, classes) # MMOCTF
        # self.fc = nn.Linear(2048 + 3072, classes) #OLIVES

        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

class Multi_EF_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_EF_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]

        # --- 2D early fusion conv
        if classifier_OCT_dims[0][-1] == 248:
            self.ef_conv = nn.Sequential(
                        nn.AvgPool2d(kernel_size=1, stride=[2,2],
                         ceil_mode=True, count_include_pad=False),
                        nn.Conv2d(3, 3, 1, 1))
            self.fc = nn.Linear(3584, classes)

        else:
            self.ef_conv = nn.Sequential(
                        nn.AvgPool2d(kernel_size=1, stride=[4,2],
                         ceil_mode=True, count_include_pad=False),
                        nn.Conv2d(3, 3, 1, 1))
            self.fc = nn.Linear(8704, classes)

        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        X0_features = self.ef_conv(X[0])
        if self.classes == 2:
            if X[1].shape[-1] == 248:
                X[1].resize_(X[1].shape[0],X[1].shape[1],X[1].shape[2],X0_features.shape[-2],X0_features.shape[-1])
                combine_features = torch.cat([X0_features.unsqueeze(1),X[1].permute(0,1,2,4,3)],2)

            else:
                combine_features = torch.cat([X0_features.unsqueeze(1),X[1].permute(0,1,2,4,3)],2)
        else:
            combine_features = torch.cat([X0_features.unsqueeze(1),X[1]],2)

        # backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(combine_features)
        pred = self.fc(backboneout_2)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss


class CBAM2D(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM2D, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class CBAM3D(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM3D, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Multi_CBAM_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_CBAM_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_feature_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]

        # ---- CBAM Layer----
        self.CBAM2D_layer = CBAM2D(2048)
        self.CBAM3D_layer = CBAM3D(512)
        # GAP
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.resnet_3DNet = Medical_feature_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        if classifier_OCT_dims[0][0] == 128:
            self.fc = nn.Linear(2048 + 8192, classes) # MMOCTF
        else:
            self.fc = nn.Linear(2048 + 3072, classes) # OLIVES

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        backboneout_1_CBAM = self.CBAM2D_layer(backboneout_1)
        backboneout_2_CBAM = self.CBAM3D_layer(backboneout_2)
        backboneout_1_CBAM_GAP = self.avgpool(backboneout_1_CBAM)
        backboneout_2_CBAM_GAP = self.avgpool(backboneout_2_CBAM)
        backboneout_1_CBAM_GAP = backboneout_1_CBAM_GAP.view(backboneout_1_CBAM_GAP.size(0), -1)
        backboneout_2_CBAM_GAP = backboneout_2_CBAM_GAP.view(backboneout_2_CBAM_GAP.size(0), -1)
        combine_features = torch.cat([backboneout_1_CBAM_GAP,backboneout_2_CBAM_GAP],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss


class Multi_ensemble_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_ensemble_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base2_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.fc = nn.Linear(2048 + 8192, classes)
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

class Multi_ensemble_3D_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_ensemble_3D_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base2_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.fc = nn.Linear(2048 + 8192, classes)
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

class Multi_dropout_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_dropout_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_dropout_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_dropout_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()

        if classifier_OCT_dims[0][0] == 128:
            self.fc = nn.Linear(2048 + 8192, classes) # MMOCTF
        else:
            self.fc = nn.Linear(2048 + 3072, classes) # OLIVES

        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss