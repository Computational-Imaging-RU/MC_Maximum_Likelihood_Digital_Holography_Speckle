import argparse
import time
import cv2
import os
import glob
import pickle
import random
from skimage.metrics import structural_similarity as ssim
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import torch
import torch.nn.functional as F
import numpy as np
import math

from utils import gen_latent_code_patch, PSNR
from PGD_MC_complex import nll_grad_operator_MC_CGD
from decoder import autoencodernet
from train_DnCNN_origin import DnCNN #remember to change this when using different DnCNN

from bm3d import bm3d, BM3DStages


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

    return aperture, scaling_factor


def train(aperture, scaling_factor, out_path, filepaths, channel_list, dtype, device):

    img_te_num = len(filepaths)
    ########## Save the running logs ##########
    PSNR_GD_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    PSNR_NN_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    SSIM_GD_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    SSIM_NN_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    CG_iter_1_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    CG_iter_2_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)

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
            # xw = torch.mul(img_gt_, w_noise)
            xw_arr = xw.detach().cpu().numpy()

            img_fft = fftshift(fft2(xw_arr)) # normalize the DFT, shift DC to center
            img_fft_aperture = img_fft * aperture # apply centered mask
            img_blur_l = ifft2(ifftshift(img_fft_aperture)) # unshift to corner

            # img_fft = fftshift(fft2(xw_arr, norm="ortho")) # normalize the DFT, shift DC to center
            # img_fft_aperture = img_fft * aperture # apply centered mask
            # img_blur_l = ifft2(ifftshift(img_fft_aperture), norm="ortho") # unshift to corner


            # compute the SNR
            noise_power_theory = args.add_std**2
            sig_power = np.abs(img_blur_l)**2
            sig_power_mean = np.mean(sig_power)
            db_inter = sig_power_mean / (noise_power_theory + 1e-12)
            ratio_db_theory = 10*np.log10(db_inter + 1e-12)
            print("ratio_db_theory =", ratio_db_theory)


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

        x_new = np.clip(x_init, 0, 1)
        y = img_blur

        total_start_time = time.time()
        ###########################################################
        ################### Iterative PGD ####################
        ###########################################################
        for outer_idx in range(args.outer_ite):
            print('outer ite:', outer_idx + 1)

            ###########################################################
            ################### GD step: ##########################
            ###########################################################
            GD_start_time = time.time()
            grad_matrix, CG_iter_1, CG_iter_2 = nll_grad_operator_MC_CGD(x_new, y, aperture, args.add_std_prime, args.num_ite_MC)
            x_G = x_new - args.lr_GD * grad_matrix
            x_G_save = np.clip(x_G, 0, 1) * 255.0
            psnr_GD = PSNR(x_G_save, img_gt * 255.0)
            ssim_GD = ssim(x_G_save, img_gt * 255.0, data_range=255)
            print('psnr GD', psnr_GD, 'ssim GD', ssim_GD)
            GD_end_time = time.time()

            ###########################################################
            ################### projection step: ####################
            ###########################################################
            projection_start = time.time()
            x_raw = torch.from_numpy(x_G).type(dtype).to(device)

            ######### train Deep Decoder ##########
            if args.denoiser == 'DIP':
                output_depth = 1 # number of output channels (gray scale image)
                DIP_patch_size = patch_size
                net = autoencodernet(num_output_channels=output_depth, 
                                    num_channels_up=channel_list,
                                    need_sigmoid=args.out_nonlinear, 
                                    decodetype=args.decodetype,
                                    kernel_size=args.kernel_size).type(dtype).to(device)
                latent_code = gen_latent_code_patch(1, DIP_patch_size, channel_list, 1).type(dtype).to(device)
                params = [x for x in net.decoder.parameters()]
                optimizer = torch.optim.Adam(params, lr=args.lr_NN, weight_decay=args.weight_decay)
                for ee in range(args.inner_ite):
                    net.train()
                    optimizer.zero_grad()
                    x_gen_tensor = net(latent_code).squeeze(0).squeeze(0)
                    loss_train = F.mse_loss(x_gen_tensor, x_raw.detach())
                    loss_train.backward()
                    optimizer.step()
                with torch.no_grad():
                    x_gen = net(latent_code).squeeze(0).squeeze(0).detach()
                    x_gen = x_gen.detach().cpu().numpy()

            ########## Load pre-trained denoiser: DnCNN ##########
            elif args.denoiser == 'DnCNN':
                model_path = "checkpoints_dncnn_origin/" + f"17_64_True_128_40_320_{args.DnCNN_sigma}_{args.denoiser_loss}/" + "dncnn_best.pth"
                ckpt = torch.load(model_path, map_location=args.device)
                args_DnCNN = ckpt["args"]
                denoiser = DnCNN(channels=1, 
                                 layers=args_DnCNN["layers"], 
                                 features=args_DnCNN["features"])
                denoiser.load_state_dict(ckpt["model"])
                denoiser.to(device).eval()

                x_raw_tensor = torch.as_tensor(x_raw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    x_gen_tensor, _ = denoiser(x_raw_tensor)
                x_gen = x_gen_tensor.squeeze().detach().cpu().numpy()
                x_gen = np.clip(x_gen, 0.0, 1.0)

            ########## Use denoiser: BM3D ##########
            elif args.denoiser == 'BM3D':
                x_raw = x_raw.detach().cpu().numpy()
                x_raw = np.clip(x_raw, 0.0, 1.0)
                stage_map = {"all": BM3DStages.ALL_STAGES,
                             "hard": BM3DStages.HARD_THRESHOLDING,
                             "wiener": BM3DStages.WIENER_FILTERING,}
                stage_arg = stage_map.get("all", BM3DStages.ALL_STAGES)
                x_gen = bm3d(x_raw, sigma_psd=args.BM3D_sigma, stage_arg=stage_arg)
                x_gen = np.clip(x_gen, 0.0, 1.0).astype(np.float32)

            projection_end = time.time()

            with torch.no_grad():
                # net.eval()
                x_new = x_gen

                x_gen_save = np.clip(x_gen, 0, 1) * 255.0
                psnr_NN = PSNR(x_gen_save, img_gt * 255.0)
                ssim_NN = ssim(x_gen_save, img_gt * 255.0, data_range=255)
                print('psnr NN', psnr_NN, 'ssim NN', ssim_NN)
                print('GD time', GD_end_time - GD_start_time, 'proj time', projection_end - projection_start)

                # Save the results and reconstructed images
                PSNR_GD_All[outer_idx+1, img_no] = psnr_GD
                SSIM_GD_All[outer_idx+1, img_no] = ssim_GD
                PSNR_NN_All[outer_idx+1, img_no] = psnr_NN
                SSIM_NN_All[outer_idx+1, img_no] = ssim_NN
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_GD_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, outer_idx, psnr_GD, ssim_GD)), x_G_save.round().astype(np.uint8))
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_NN_PSNR_%.3f_SSIM_%.5f.png" % (single_imgName, outer_idx, psnr_NN, ssim_NN)), x_gen_save.round().astype(np.uint8))

                def stats(name, x):
                    print(f"{name}: min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} std={x.std():.4f}")
                stats("x_G", x_G); stats("x_gen", x_gen)

                CG_iter_1_All[outer_idx, img_no] = CG_iter_1
                CG_iter_2_All[outer_idx, img_no] = CG_iter_2

        total_end_time = time.time()
        print('total running time:', total_end_time - total_start_time)

    return PSNR_GD_All, SSIM_GD_All, PSNR_NN_All, SSIM_NN_All, CG_iter_1_All, CG_iter_2_All

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
    parser.add_argument('--dataset', type=str, default='Set11_peppers', help='test dataset')
    parser.add_argument('--seed', type=int, default=312, help='name of test set')
    
    parser.add_argument('--mask_rate', type=float, default=0.8, help='aperture percentage')
    parser.add_argument("--aperture_donut", action="store_true", help="if use donut aperture.")
    parser.add_argument('--num_look', type=int, default=1, help='number of looks')
    parser.add_argument('--add_std', type=float, default=0.2, help='additive noise standard deviation.')
    parser.add_argument('--add_std_prime', type=float, default=0.2, help='additive noise standard deviation used in algorithm.')

    parser.add_argument('--x_init', type=str, default='AHy_avg', help='init of PGD')
    parser.add_argument('--lr_GD', type=float, default=1e-2, help='PGD step size.')
    parser.add_argument('--outer_ite', type=int, default=50, help='PGD iterations')
    parser.add_argument('--num_ite_MC', type=int, default=20, help='num ite in MC')
    parser.add_argument('--denoiser', type=str, default='DIP', help='DIP, DnCNN, BM3D')
    
    parser.add_argument('--BM3D_sigma', type=float, default=0.3, help='sigma in BM3D.')

    parser.add_argument('--denoiser_loss', type=str, default='mse', help='loss in pre-training DnCNN')
    parser.add_argument('--DnCNN_sigma', type=float, default=125.0, help='sigma in pre-training DnCNN.')

    parser.add_argument('--kernel_size', type=int, default=1, help='kernel size in DIP')
    parser.add_argument('--decodetype', type=str, default='upsample', help='upsample, transposeconv')
    parser.add_argument('--out_nonlinear', type=bool, default=True, help='DIP output layer')
    parser.add_argument('--lr_NN', type=float, default=1e-3, help='DIP learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for DIP training.')
    parser.add_argument('--inner_ite', type=int, default=100, help='DIP training iterations')

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
    channel_list = [100,50,25,10]
    # channel_list = [200,100,50,25,10]
    # channel_list = [128,128,128,128]

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
    out_path = os.path.join('./results_PGD_MC_complex_test', "_".join(map(str, [args.dataset, args.seed,
                                                                   args.mask_rate, args.aperture_donut,
                                                                   args.num_look, args.add_std,
                                                                   args.add_std_prime,
                                                                   args.x_init, args.lr_GD,
                                                                   args.outer_ite, args.num_ite_MC,
                                                                   args.denoiser,
                                                                   args.BM3D_sigma,
                                                                   args.denoiser_loss, args.DnCNN_sigma,
                                                                   args.kernel_size, args.decodetype,
                                                                   args.out_nonlinear,
                                                                   args.lr_NN, args.weight_decay,
                                                                   args.inner_ite, channel_list])))
    os.makedirs(out_path, exist_ok=True)
    filepaths = glob.glob(os.path.join(args.data_dir, args.dataset) + '/*.png')

    ############# training function #############
    PSNR_GD_All, SSIM_GD_All, PSNR_NN_All, SSIM_NN_All, CG_iter_1_All, CG_iter_2_All= train(aperture, scaling_factor, out_path, filepaths, channel_list, dtype, device)

    with open(out_path + '/' + 'PSNR_GD' + '.pkl', 'wb') as psnr_GD_file:
        pickle.dump(PSNR_GD_All, psnr_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_GD' + '.pkl', 'wb') as ssim_GD_file:
        pickle.dump(SSIM_GD_All, ssim_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'PSNR_NN' + '.pkl', 'wb') as psnr_NN_file:
        pickle.dump(PSNR_NN_All, psnr_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_NN' + '.pkl', 'wb') as ssim_NN_file:
        pickle.dump(SSIM_NN_All, ssim_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'CG_1' + '.pkl', 'wb') as CG_1_file:
        pickle.dump(CG_iter_1_All, CG_1_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'CG_2' + '.pkl', 'wb') as CG_2_file:
        pickle.dump(CG_iter_2_All, CG_2_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

