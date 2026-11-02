"""Redes da CycleGAN (gerador ResNet, discriminador PatchGAN) e E/S de imagem."""
from pathlib import Path

import torch
from PIL import Image
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.f(x)


class ResNetGenerator(nn.Module):
    """c7s1-64, d128, d256, n_blocks x R256, u128, u64, c7s1-3 (tanh)."""

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        L = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):  # downsampling
            L += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        L += [ResBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsampling
            L += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        L += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_ch, 7), nn.Tanh()]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN 70x70: C64 (sem norm) - C128 - C256 - C512 (stride 1) - conv 1 canal. Sem sigmoid (LSGAN)."""

    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        L = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers + 1):
            out = min(ndf * 2 ** i, ndf * 8)
            stride = 2 if i < n_layers else 1
            L += [nn.Conv2d(ch, out, 4, stride, 1), nn.InstanceNorm2d(out), nn.LeakyReLU(0.2, True)]
            ch = out
        L += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def init_weights(net: nn.Module, std: float = 0.02) -> None:
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.modules.instancenorm._InstanceNorm) and m.affine:
            nn.init.normal_(m.weight, 1.0, std)
            nn.init.zeros_(m.bias)


def load_image(path, size: int) -> torch.Tensor:
    """Imagem RGB -> tensor (3, size, size) em [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return t.view(size, size, 3).permute(2, 0, 1).float() / 127.5 - 1.0


def to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor (3, H, W) em [-1, 1] -> PIL RGB uint8."""
    t = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy(), "RGB")


def load_generators(path, device="cpu"):
    """Lê G_epoca_XXX.pt (fp16) ou last.pt; devolve (G_AB, G_BA, epoca) em float32 e eval()."""
    ck = torch.load(Path(path), map_location=device, weights_only=True)
    gs = []
    for k in ("G_AB", "G_BA"):
        g = ResNetGenerator()
        g.load_state_dict(ck[k])  # copia fp16 -> fp32
        gs.append(g.float().to(device).eval())
    return gs[0], gs[1], int(ck["epoca"])
