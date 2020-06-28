import paddle.fluid as fluid
from .backbone import resnet50, resnet101, resnet152
from .modeling import AFNB, APNB
from .nn import ReLU


class ANNN(fluid.dygraph.Layer):
    def __init__(self, backbone="resnet50", pretrained=False, root=None):
        super(ANNN, self).__init__()
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained, root)
        elif backbone == "resnet101":
            self.backbone = resnet101(pretrained, root)
        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained, root)

        self.head = None

    def forward(self, x):
        imsize = x.shape[2:]
        _, _, c3, c4 = self.backbone(x)

        x = self.head(c3, c4)
        outputs = fluid.layers.resize_bilinear(input=x, out_shape=imsize)

        return tuple(outputs)


class ANNN_Head(fluid.dygraph.Layer):
    def __init__(self, out_channels, norm_layer=fluid.dygraph.BatchNorm):
        super(ANNN_Head, self).__init__()
        # low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout
        self.fusion = fluid.dygraph.Sequential(
            AFNB(1024, 2048, 2048, 256, 256, dropout=0.05, sizes=([1]), norm_layer=norm_layer),
            fluid.dygraph.Conv2D(2048, 512, filter_size=3, stride=1, padding=1),
            norm_layer(512),
            ReLU(),
        )
        # extra added layers
        self.context = APNB(in_channels=512, out_channels=512, key_channels=256, value_channels=256, dropout=0.05, sizes=([1]), norm_layer=norm_layer)
        self.cls = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(1024, 512, filter_size=3, stride=1, padding=1, bias_attr=False),
            norm_layer(512),
            ReLU(),
            fluid.dygraph.Conv2D(512, out_channels, filter_size=1, stride=1, padding=0, bias_attr=True)
        )

    def forward(self, low_feats, high_feats):
        afnb = self.fusion(low_feats, high_feats)
        apnb = self.context(afnb)
        x = self.cls(fluid.layers.concat([afnb, apnb], axis=1))
        return x
