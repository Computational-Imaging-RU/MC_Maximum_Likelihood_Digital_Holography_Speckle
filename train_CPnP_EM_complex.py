import argparse
import time
import cv2
import os
import glob
import pickle
import random
from skimage.metrics import structural_similarity as ssim
from scipy.fft import dctn, idctn, fft2, ifft2, fftshift, ifftshift
import torch
import numpy as np
import math

from utils import PSNR, gen_latent_code_patch
from EM_complex import E_step, M_Step
from train_DnCNN_origin import DnCNN #remember to change this when using different DnCNN
from decoder import autoencodernet

def create_aperture(image_height, image_width, aperture_radius):
    """
    Creates a circular aperture relative to an image, with a radius that is a fraction of original image height/2. Also
    returns a scaling factor for brightness correction.
    """
    # Define circular aperture (1 inside, 0 outside)
    if aperture_radius is None:
        # No circular aperture, set "aperture" to all ones. Mostly for testing MLE loss.
        aperture_radius = 0
        aperture = np.ones((image_height, image_width))
        scaling_factor = 1
    elif aperture_radius == "donut":
        # Specialized aperture for USAFA data. donut_radius=1
        aperture_radius = 1
        aperture_radius = int(round((image_height / 2) * aperture_radius)) # Convert to radius in pixels
        aperture = np.zeros((image_height, image_width))
        # Create the main aperture. Add +1 to fix rounding issues with cv2
        cv2.circle(aperture, (image_width//2, image_height//2), aperture_radius+1, 1, -1)
        # Draw a zero circle in the middle to create donut
        inner_radius = 0.344  # Determined from experimental data
        inner_radius = int(round((image_height / 2) * inner_radius))
        cv2.circle(aperture, (image_width // 2, image_height // 2), inner_radius, 0, -1)
        # Compute scaling factor
        total_image_area = image_height * image_width
        aperture_area = np.sum(aperture)
        scaling_factor = total_image_area / aperture_area
    else:
        aperture_radius = int(round((image_height / 2) * aperture_radius))  # Convert to radius in pixels
        print('aperture radius', aperture_radius)
        aperture = np.zeros((image_height, image_width))
        # Create the aperture. Add +1 to fix rounding issues with cv2
        cv2.circle(aperture, (image_width//2, image_height//2), aperture_radius+1, 1, -1)
        # Calculate the relative aperture area (ratio of aperture area to total image area) and brightness scaling factor
        total_image_area = image_height * image_width
        aperture_area = np.sum(aperture)
        scaling_factor = total_image_area / aperture_area

    # aperture = np.expand_dims(aperture, 0)
    # aperture = np.expand_dims(aperture, 0)
    return aperture, scaling_factor


def train(aperture, out_path, filepaths, dtype, device, args):

    img_te_num = len(filepaths)
    ########## Save the running logs ##########
    PSNR_NN_All = np.zeros([args.EM_ite+1, img_te_num], dtype=np.float64)
    SSIM_NN_All = np.zeros([args.EM_ite+1, img_te_num], dtype=np.float64)

    ########## Loop over every test image ##########
    for img_no in range(img_te_num):
        imgName = filepaths[img_no]
        single_imgName_ = imgName.split(".")[0]
        single_imgName = single_imgName_.split("/")[-1]
        print('image name:', imgName)

        ########## Prepare the image ##########
        Img = cv2.imread(imgName, 1)
        patch_size = np.shape(Img)[0]
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb) / 255.0
        img_gt_ = torch.from_numpy(Img_yuv[:, :, 0]).type(dtype).to(device)
        img_gt = img_gt_.detach().cpu().numpy()
        cv2.imwrite(os.path.join(out_path, "%s_raw.png" % (single_imgName)), (np.clip(img_gt, 0, 1)*255.0).round().astype(np.uint8))

        ########## generate the blurred (multi-look) measurements ##########
        img_blur = np.zeros((args.num_look, patch_size, patch_size), dtype=np.complex64)
        for look_idx in range(args.num_look):
            # generate complex speckle noise w
            w_real = (torch.randn(img_gt.shape) / math.sqrt(2)).to(dtype)
            w_img = (torch.randn(img_gt.shape) / math.sqrt(2)).to(dtype)
            w_noise = torch.complex(w_real, w_img).to(device)
            xw = torch.mul(torch.sqrt(img_gt_), w_noise)
            xw_arr = xw.detach().cpu().numpy()

            img_fft = fftshift(fft2(xw_arr)) # shift DC to center
            img_fft_aperture = img_fft * aperture # apply centered mask
            img_blur_l = ifft2(ifftshift(img_fft_aperture)) # unshift to corner

            z_real = np.random.normal(loc=0.0, scale=args.add_std / np.sqrt(2), size=(patch_size, patch_size))
            z_imag = np.random.normal(loc=0.0, scale=args.add_std / np.sqrt(2), size=(patch_size, patch_size))
            z_noise_ = torch.complex(torch.from_numpy(z_real).type(dtype), torch.from_numpy(z_imag).type(dtype)).to(device)
            z_noise = z_noise_.detach().cpu().numpy()
            img_blur_z_l = img_blur_l + z_noise
            Axw_z_arr = img_blur_z_l
            img_blur[look_idx] = Axw_z_arr

        ########## Init the GD input ##########
        AHy_square_sum = np.zeros((patch_size, patch_size))
        for look in range(args.num_look):
            y_i = img_blur[look]
            y_dft = fftshift(fft2(y_i))
            y_dft_aperture = y_dft * aperture
            AHy = ifft2(ifftshift(y_dft_aperture))
            AHy_square = np.square(np.abs(AHy))
            AHy_square_sum += AHy_square
        AHy_square_mean = AHy_square_sum / args.num_look

        psnr_AHy_square_mean = PSNR(np.clip(AHy_square_mean, 0, 1) * 255.0, img_gt * 255.0)
        ssim_AHy_square_mean = ssim(np.clip(AHy_square_mean, 0, 1) * 255.0, img_gt * 255.0, data_range=255)
        cv2.imwrite(os.path.join(out_path, "%s_AHy_square_mean_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, psnr_AHy_square_mean, ssim_AHy_square_mean)), (np.clip(AHy_square_mean, 0, 1) * 255.0).round().astype(np.uint8))
        print('psnr AHy_square_mean', psnr_AHy_square_mean, 'ssim AHy_square_mean', ssim_AHy_square_mean)

        if args.x_init == 'constant':
            x_init = np.ones((patch_size, patch_size)) * 0.5
        elif args.x_init == 'AHy_avg':
            x_init = AHy_square_mean
        print(f'PGD initialization:{args.x_init}')
        psnr_x_init = PSNR(np.clip(x_init, 0, 1) * 255.0, img_gt * 255.0)
        ssim_x_init = ssim(np.clip(x_init, 0, 1) * 255.0, img_gt * 255.0, data_range=255)
        cv2.imwrite(os.path.join(out_path, "%s_x_init_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, psnr_x_init, ssim_x_init)), (np.clip(x_init, 0, 1) * 255.0).round().astype(np.uint8))
        print('psnr init', psnr_x_init, 'ssim init', ssim_x_init)
        PSNR_NN_All[0, img_no] = psnr_x_init
        SSIM_NN_All[0, img_no] = ssim_x_init

        r1_init, r2_init, r_bar_prime_init = np.clip(x_init,0,1), np.clip(x_init,0,1), np.clip(x_init,0,1)
        y = img_blur

        ########## Load pre-trained denoiser ##########
        if args.denoiser == 'DnCNN':
            model_path = "checkpoints_dncnn_origin/" + f"17_64_True_128_40_320_{args.std_F2}_{args.denoiser_loss}/" + "dncnn_best.pth"
            ckpt = torch.load(model_path, map_location=args.device)
            args_DnCNN = ckpt["args"]
            denoiser = DnCNN(channels=1, layers=args_DnCNN["layers"], features=args_DnCNN["features"])
            denoiser.load_state_dict(ckpt["model"])
            denoiser.to(device).eval()

        total_start_time = time.time()
        ########## EM ##########
        r_in_1, r_in_2, r_bar_prime = r1_init, r2_init, r_bar_prime_init
        for em_idx in range(args.EM_ite):
            print('EM ite:', em_idx + 1)

            ########## Load un-trained denoiser ##########
            if args.denoiser == 'DIP':
                net = autoencodernet(num_output_channels=1, num_channels_up=[100,50,25,10],
                                     need_sigmoid=True, decodetype='upsample',
                                     kernel_size=args.kernel_size).type(dtype).to(device)
                latent_code = gen_latent_code_patch(1, patch_size, [100,50,25,10], 1).type(dtype).to(device)
                params = [x for x in net.decoder.parameters()]
                optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.0)
                denoiser = [net, latent_code, optimizer]
            elif args.denoiser == 'BM3D':
                denoiser = args.std_F2 / 255.0

            E_start_time = time.time()
            mu_square, C = E_step(r_bar_prime, y, aperture, args.add_std_prime)
            E_end_time = time.time()
            print('E step time:', E_end_time - E_start_time)
            r_bar_prime, r_F1, r_F2, r_W1, r_W2 = M_Step(r_in_1, r_in_2, mu_square, C, args.rho, denoiser, np.sqrt(args.std_F1), args.denoiser, args.DIP_ite)
            r_in_1, r_in_2 = r_F1, r_F2
            with torch.no_grad():
                x_F1 = np.clip(r_in_1, 0, 1) * 255.0
                psnr_F1 = PSNR(x_F1, img_gt * 255.0)
                ssim_F1 = ssim(x_F1, img_gt * 255.0, data_range=255)
                print('psnr F1', psnr_F1, 'ssim F1', ssim_F1)
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_F1_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, em_idx, psnr_F1, ssim_F1)), x_F1.round().astype(np.uint8))

                x_F2 = np.clip(r_in_2, 0, 1) * 255.0
                psnr_F2 = PSNR(x_F2, img_gt * 255.0)
                ssim_F2 = ssim(x_F2, img_gt * 255.0, data_range=255)
                print('psnr F2', psnr_F2, 'ssim F2', ssim_F2)
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_F2_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, em_idx, psnr_F2, ssim_F2)), x_F2.round().astype(np.uint8))

                x_W1 = np.clip(r_W1, 0, 1) * 255.0
                psnr_W1 = PSNR(x_W1, img_gt * 255.0)
                ssim_W1 = ssim(x_W1, img_gt * 255.0, data_range=255)
                print('psnr W1', psnr_W1, 'ssim W1', ssim_W1)
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_W1_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, em_idx, psnr_W1, ssim_W1)), x_W1.round().astype(np.uint8))

                x_W2 = np.clip(r_W2, 0, 1) * 255.0
                psnr_W2 = PSNR(x_W2, img_gt * 255.0)
                ssim_W2 = ssim(x_W2, img_gt * 255.0, data_range=255)
                print('psnr W2', psnr_W2, 'ssim W2', ssim_W2)
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_W2_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, em_idx, psnr_W2, ssim_W2)), x_W2.round().astype(np.uint8))

                x_gen_save = np.clip(r_bar_prime, 0, 1) * 255.0
                psnr_NN = PSNR(x_gen_save, img_gt * 255.0)
                ssim_NN = ssim(x_gen_save, img_gt * 255.0, data_range=255)
                print('psnr NN', psnr_NN, 'ssim NN', ssim_NN)
                # Save the results and reconstructed images
                PSNR_NN_All[em_idx+1, img_no] = psnr_NN
                SSIM_NN_All[em_idx+1, img_no] = ssim_NN
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_NN_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, em_idx, psnr_NN, ssim_NN)), x_gen_save.round().astype(np.uint8))

                def stats(name, x):
                    print(f"{name}: min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} std={x.std():.4f}")
                stats("r1", r_in_1); stats("r2", r_in_2); stats("r_bar_prime", r_bar_prime)

        total_end_time = time.time()
        print('total running time:', total_end_time - total_start_time)

    return PSNR_NN_All, SSIM_NN_All

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
    parser.add_argument('--dataset', type=str, default='Set11_peppers', help='test dataset')
    parser.add_argument('--seed', type=int, default=312, help='name of test set')

    parser.add_argument('--mask_rate', type=float, default=0.5, help='aperture percentage')
    parser.add_argument("--aperture_donut", action="store_true", help="if use donut aperture.")
    parser.add_argument('--num_look', type=int, default=1, help='number of looks')
    parser.add_argument('--add_std', type=float, default=0.2, help='additive noise standard deviation.')
    parser.add_argument('--add_std_prime', type=float, default=0.2, help='additive noise standard deviation used in algorithm.')

    parser.add_argument('--x_init', type=str, default='AHy_avg', help='init for EM')
    parser.add_argument('--EM_ite', type=int, default=50, help='EM iterations')

    parser.add_argument('--denoiser', type=str, default='DIP', help='denoiser in F2 (denoiser)')
    parser.add_argument('--kernel_size', type=int, default=1, help='kernel size in DIP')
    parser.add_argument('--DIP_ite', type=int, default=50, help='number of training iteration for DIP')
    parser.add_argument('--denoiser_loss', type=str, default='mse', help='loss used for training denoiser')

    parser.add_argument('--std_F1', type=float, default=0.1, help='noise standard deviation in F1 proximal.')
    parser.add_argument('--std_F2', type=float, default=25.0, help='noise sd in F2 proximal (DnCNN, BM3D).')
    parser.add_argument('--rho', type=float, default=0.2, help='Mann iteration coefficient.')
    args = parser.parse_args()
    print(args)

    ############# Initialize the random seed ##############
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dtype = torch.float64

    filepaths = glob.glob(os.path.join(args.data_dir, args.dataset) + '/*.png')
    imgName = filepaths[0]
    Img = cv2.imread(imgName, 1)
    height, width = np.shape(Img)[0], np.shape(Img)[1]
    if args.aperture_donut:
        aperture_radius = 'donut'
    else:
        aperture_radius = args.mask_rate
    aperture, scaling_factor = create_aperture(height, width, aperture_radius)
    transparency_ratio = 1.0 / scaling_factor
    print('aperture transparency ratio:', transparency_ratio)

    ############# testing data and saving path #############
    out_path = os.path.join('./results_CPnP_EM_complex', "_".join(map(str, [args.dataset, args.mask_rate,
                                                                    args.aperture_donut,
                                                                    args.num_look, args.add_std,
                                                                    args.add_std_prime,
                                                                    args.x_init, args.EM_ite,
                                                                    args.denoiser, args.kernel_size,
                                                                    args.DIP_ite,
                                                                    args.denoiser_loss,
                                                                    args.std_F1, args.std_F2,
                                                                    args.rho])))
    os.makedirs(out_path, exist_ok=True)
    filepaths = glob.glob(os.path.join(args.data_dir, args.dataset) + '/*.png')

    ############# training function #############
    PSNR_NN_All, SSIM_NN_All = train(aperture, out_path, filepaths, dtype, device, args)

    with open(out_path + '/' + 'PSNR_NN' + '.pkl', 'wb') as psnr_NN_file:
        pickle.dump(PSNR_NN_All, psnr_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_NN' + '.pkl', 'wb') as ssim_NN_file:
        pickle.dump(SSIM_NN_All, ssim_NN_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

