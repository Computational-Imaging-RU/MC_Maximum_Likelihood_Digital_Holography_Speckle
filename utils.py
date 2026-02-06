import numpy as np
import math
import torch


def gen_latent_code(num_frame, num_channels, patch_size, out_ch):
    # img_meas, A_sensing are all tensor
    totalupsample = 2 ** (len(num_channels) - 1)
    # if running compressive imaging
    w = np.sqrt(int(patch_size**2 / out_ch))
    width = int(w / (totalupsample))
    height = int(w / (totalupsample))
    # (1, num_channel_init, width_init, height_init)
    # shape = [num_frame, num_channels[0], width, height]
    shape = [1, num_channels[0], width, height]
    print("shape of latent code: ", shape)
    # latent_code = nn.Parameter(torch.zeros(shape))
    latent_code = torch.zeros(shape)
    latent_code.data.normal_()
    latent_code.data *= 1. / 10
    return latent_code

def gen_latent_code_patch(batch_size, patch_size, num_channels, out_ch):
  # img_meas, A_sensing are all tensor
  totalupsample = 2 ** (len(num_channels) - 1)
  # if running as decoder/compressor
  width, height = 0, 0

  w = patch_size
  width = int(w / (totalupsample))
  height = int(w / (totalupsample))

  # (1, num_channel_init, width_init, height_init)
  shape = [batch_size, num_channels[0], width, height]
  # print("shape of latent code: ", shape)
  # latent_code = nn.Parameter(torch.zeros(shape))
  latent_code = torch.zeros(shape)
  latent_code.data.normal_()
  latent_code.data *= 1. / 10
  return latent_code

def PSNR(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    # print('img1', img1)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
