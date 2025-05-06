import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        raise ValueError("norm_type must be bn or gn")

class ReparamConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deploy=False):
        super(ReparamConvBlock, self).__init__()
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        else:
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.branch_identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        if self.deploy:
            return self.reparam_conv(x)
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.branch_identity is not None:
            out += self.branch_identity(x)
        return out

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.branch_3x3[0].in_channels,
            out_channels=self.branch_3x3[0].out_channels,
            kernel_size=3,
            stride=self.branch_3x3[0].stride,
            padding=self.branch_3x3[0].padding,
            bias=True
        ).to(device)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        self.deploy = True
        del self.branch_3x3, self.branch_1x1
        if self.branch_identity is not None:
            del self.branch_identity

    def _fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        fused_conv_w = conv.weight * t.reshape(-1, 1, 1, 1)
        fused_conv_b = bn.bias - bn.running_mean * bn.weight / std
        return fused_conv_w, fused_conv_b

    def _get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        ksize = self.branch_3x3[0].kernel_size[0]
        pad = ksize // 2
        k1_padded = F.pad(k1, [pad, pad, pad, pad])


        if self.branch_identity is not None:
            device = self.branch_identity.weight.device
            identity_conv = nn.Conv2d(in_channels=self.branch_identity.num_features, out_channels=self.branch_identity.num_features, kernel_size=1,
            stride=1, padding=0, bias=False).to(device)
            identity_conv.weight.data = torch.eye(identity_conv.out_channels, device=device).view(identity_conv.out_channels, identity_conv.out_channels, 1, 1)
            k_id, b_id = self._fuse_conv_bn(identity_conv, self.branch_identity)
            k_id = F.pad(k_id, [pad, pad, pad, pad])
        else:
            k_id, b_id = 0, 0

        kernel = k3 + k1_padded + (k_id if isinstance(k_id, torch.Tensor) else 0)
        bias = b3 + b1 + (b_id if isinstance(b_id, torch.Tensor) else 0)
        return kernel, bias


class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        mid_channels = channel_out // 2
        total_channels = mid_channels + channel_out
        self.conv1 = ReparamConvBlock(channel_in, total_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.norm2 = get_norm_layer(mid_channels, norm_type=norm_type)
        self.conv2 = ReparamConvBlock(mid_channels, channel_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResUp(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        total_channels = (channel_in // 2) + channel_out
        self.conv1 = ReparamConvBlock(channel_in, total_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)
        self.conv2 = ReparamConvBlock(channel_in // 2, channel_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        if channel_in == channel_out:
            first_out = channel_in // 2
            self.skip = True
        else:
            first_out = (channel_in // 2) + channel_out
            self.skip = False
        self.conv1 = ReparamConvBlock(channel_in, first_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        bottleneck = channel_in // 2
        self.norm2 = get_norm_layer(bottleneck, norm_type=norm_type)
        self.conv2 = ReparamConvBlock(bottleneck, channel_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.bttl_nk = bottleneck

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))
        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]
        skip = x_in if self.skip else x_cat[:, self.bttl_nk:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class Encoder(nn.Module):
    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=250, num_res_blocks=1, norm_type="bn", deep_model=False):
        super(Encoder, self).__init__()
        self.conv_in = ReparamConvBlock(channels, blocks[0] * ch, kernel_size=3, stride=1, padding=1)
        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]
        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):
            if deep_model:
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))
        self.conv_mu = ReparamConvBlock(widths_out[-1] * ch, latent_channels, kernel_size=1, stride=1, padding=0)
        self.conv_log_var = ReparamConvBlock(widths_out[-1] * ch, latent_channels, kernel_size=1, stride=1, padding=0)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = x.to(next(self.parameters()).device)  # Ensure input is on the correct device
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        x = self.sample(mu, log_var) if self.training or sample else mu
        return x, mu, log_var


class Decoder(nn.Module):
    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=250, num_res_blocks=1, norm_type="bn", deep_model=False):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]
        self.conv_in = ReparamConvBlock(latent_channels, widths_in[0] * ch, kernel_size=1, stride=1, padding=0)
        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))
        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))
        self.conv_out = ReparamConvBlock(blocks[0] * ch, channels, kernel_size=5, stride=1, padding=2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = x.to(next(self.parameters()).device)  # Ensure input is on the correct device
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        return torch.tanh(self.conv_out(x))


class VAE(nn.Module):
    def __init__(self, channel_in=3, ch=64, blocks=(1, 2, 4, 8), latent_channels=251, num_res_blocks=1, norm_type="bn", deep_model=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels, num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels, num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var

    def convert_to_deploy(self):
        for module in self.modules():
            if isinstance(module, ReparamConvBlock):
                module.switch_to_deploy()
