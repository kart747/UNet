# U-Net (PyTorch)

A from-scratch PyTorch implementation of the U-Net architecture for semantic
segmentation, written while learning the paper
[*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)
(Ronneberger et al., 2015).

## Architecture

```
input ─ DoubleConv ─────────────────────────── skip ──▶ concat ─ DoubleConv ─ 1x1 conv ─ output
            │ pool                                          ▲ upsample
        DoubleConv ──────────────────── skip ──▶ concat ─ DoubleConv
            │ pool                                  ▲ upsample
          ...                                     ...
            └────────────── bottleneck ─────────────┘
```

- **Encoder:** `DoubleConv` followed by 2x2 max pooling, repeated for each of
  the `features` (default `64, 128, 256, 512`). Height and width halve, channels grow.
- **Bottleneck:** one `DoubleConv` at the lowest resolution (`1024` channels by default).
- **Decoder:** `ConvTranspose2d` doubles height and width, the matching encoder
  output (the *skip connection*) is concatenated along the channel dimension,
  then a `DoubleConv` fuses the two.
- **Head:** a 1x1 convolution maps the final features to the number of classes.

`DoubleConv` is `(Conv3x3 -> BatchNorm -> ReLU)` twice, with `padding=1` so the
spatial size is preserved.

### Differences from the original paper

- Convolutions use padding, so feature maps are not cropped before concatenation.
- BatchNorm is added after each convolution.
- If the input size is odd, the upsampled map is resized to match its skip
  connection (see `TF.resize` in `UNet.forward`).

## Requirements

- Python 3.9+
- `torch`
- `torchvision`

## Usage

```python
import torch
from unet import UNet

model = UNet(in_channels=3, out_channels=1)
x = torch.randn(1, 3, 256, 256)
y = model(x)
print(y.shape)  # torch.Size([1, 1, 256, 256])
```

The output has the same height and width as the input, with one channel per
class (raw scores, no activation applied).

## Run the check

```
python unet.py
```

This builds the model, runs a random `3 x 1 x 161 x 161` batch through it, asserts
that the output shape matches the input shape, and prints `OK`.

## Files

- `unet.py`: `DoubleConv` and `UNet` models, plus a shape test.
