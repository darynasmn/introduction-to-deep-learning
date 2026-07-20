"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp

        in_c = hp["in_channels"]
        num_classes = hp["num_classes"]

        c1, c2, c3, c4, cb = hp["c1"], hp["c2"], hp["c3"], hp["c4"], hp["cb"]
        k = hp["kernel_size"]
        p = hp["padding"]
        bias = hp["use_bias"]

        self.norm_type = hp["norm"]
        self.align_corners = hp["align_corners"]
        drop_p = hp["dropout"]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=self.align_corners)

        self.dropout_dec2 = nn.Dropout2d(p=drop_p) if drop_p > 0 else nn.Identity()
        self.dropout_dec1 = nn.Dropout2d(p=drop_p) if drop_p > 0 else nn.Identity()

        def N(ch):
            return nn.BatchNorm2d(ch) if self.norm_type == "batch" else nn.Identity()

        self.enc1_conv1 = nn.Conv2d(in_c, c1, k, padding=p, bias=bias)
        self.enc1_bn1   = N(c1)
        self.enc1_conv2 = nn.Conv2d(c1, c1, k, padding=p, bias=bias)
        self.enc1_bn2   = N(c1)

        self.enc2_conv1 = nn.Conv2d(c1, c2, k, padding=p, bias=bias)
        self.enc2_bn1   = N(c2)
        self.enc2_conv2 = nn.Conv2d(c2, c2, k, padding=p, bias=bias)
        self.enc2_bn2   = N(c2)

        self.enc3_conv1 = nn.Conv2d(c2, c3, k, padding=p, bias=bias)
        self.enc3_bn1   = N(c3)
        self.enc3_conv2 = nn.Conv2d(c3, c3, k, padding=p, bias=bias)
        self.enc3_bn2   = N(c3)

        self.enc4_conv1 = nn.Conv2d(c3, c4, k, padding=p, bias=bias)
        self.enc4_bn1   = N(c4)
        self.enc4_conv2 = nn.Conv2d(c4, c4, k, padding=p, bias=bias)
        self.enc4_bn2   = N(c4)

        self.bott_conv1 = nn.Conv2d(c4, cb, k, padding=p, bias=bias)
        self.bott_bn1   = N(cb)
        self.bott_conv2 = nn.Conv2d(cb, cb, k, padding=p, bias=bias)
        self.bott_bn2   = N(cb)

        self.dec4_conv1 = nn.Conv2d(cb + c4, c4, k, padding=p, bias=bias)
        self.dec4_bn1   = N(c4)
        self.dec4_conv2 = nn.Conv2d(c4, c4, k, padding=p, bias=bias)
        self.dec4_bn2   = N(c4)

        self.dec3_conv1 = nn.Conv2d(c4 + c3, c3, k, padding=p, bias=bias)
        self.dec3_bn1   = N(c3)
        self.dec3_conv2 = nn.Conv2d(c3, c3, k, padding=p, bias=bias)
        self.dec3_bn2   = N(c3)

        self.dec2_conv1 = nn.Conv2d(c3 + c2, c2, k, padding=p, bias=bias)
        self.dec2_bn1   = N(c2)
        self.dec2_conv2 = nn.Conv2d(c2, c2, k, padding=p, bias=bias)
        self.dec2_bn2   = N(c2)

        self.dec1_conv1 = nn.Conv2d(c2 + c1, c1, k, padding=p, bias=bias)
        self.dec1_bn1   = N(c1)
        self.dec1_conv2 = nn.Conv2d(c1, c1, k, padding=p, bias=bias)
        self.dec1_bn2   = N(c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        e1 = F.relu(self.enc1_bn1(self.enc1_conv1(x)), inplace=True)
        e1 = F.relu(self.enc1_bn2(self.enc1_conv2(e1)), inplace=True)
        p1 = self.pool(e1)

        e2 = F.relu(self.enc2_bn1(self.enc2_conv1(p1)), inplace=True)
        e2 = F.relu(self.enc2_bn2(self.enc2_conv2(e2)), inplace=True)
        p2 = self.pool(e2)

        e3 = F.relu(self.enc3_bn1(self.enc3_conv1(p2)), inplace=True)
        e3 = F.relu(self.enc3_bn2(self.enc3_conv2(e3)), inplace=True)
        p3 = self.pool(e3)

        e4 = F.relu(self.enc4_bn1(self.enc4_conv1(p3)), inplace=True)
        e4 = F.relu(self.enc4_bn2(self.enc4_conv2(e4)), inplace=True)
        p4 = self.pool(e4)

        b = F.relu(self.bott_bn1(self.bott_conv1(p4)), inplace=True)
        b = F.relu(self.bott_bn2(self.bott_conv2(b)), inplace=True)

        d4 = self.up(b)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = F.relu(self.dec4_bn1(self.dec4_conv1(d4)), inplace=True)
        d4 = F.relu(self.dec4_bn2(self.dec4_conv2(d4)), inplace=True)

        d3 = self.up(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = F.relu(self.dec3_bn1(self.dec3_conv1(d3)), inplace=True)
        d3 = F.relu(self.dec3_bn2(self.dec3_conv2(d3)), inplace=True)

        d2 = self.up(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.relu(self.dec2_bn1(self.dec2_conv1(d2)), inplace=True)
        d2 = F.relu(self.dec2_bn2(self.dec2_conv2(d2)), inplace=True)
        d2 = self.dropout_dec2(d2)

        d1 = self.up(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = F.relu(self.dec1_bn1(self.dec1_conv1(d1)), inplace=True)
        d1 = F.relu(self.dec1_bn2(self.dec1_conv2(d1)), inplace=True)
        d1 = self.dropout_dec1(d1)

        x = self.head(d1)
        return x

    def save(self, path):
        print("Saving model... %s" % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):
    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1
        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()


if __name__ == "__main__":
    pass
