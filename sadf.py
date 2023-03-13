import pyiqa
import torch

# list all available metrics
print(pyiqa.list_models())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fr_names = [
    # "AHIQ",
    "PieAPP",
    "LPIPS",
    "DISTS",
    "FSIM",
    "SSIM",
    # "MS_SSIM",
    "CW_SSIM",
    "PSNR",
    "VIF",
]
nr_names = [
    "brisque",
    "niqe",
    "ilniqe",
    "nrqm",
    "pi",
    "nima",
    "paq2piq",
    "cnniqa",
    "dbcnn",
    "musiq-ava",
    "musiq-koniq",
    "musiq-paq2piq",
    "musiq-spaq",
    # "maniqa",
    # "clipiqa",
    # "clipiqa+",
    "tres-koniq",
    "tres-flive",
]

import cv2

# Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
width = 128
# x = cv2.resize(cv2.cvtColor(cv2.imread("./images/dog.jpeg"), cv2.COLOR_BGR2RGB), (width, width))
# x = torch.tensor(x, dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
# x = x[None, ...]
#
# y = cv2.resize(cv2.cvtColor(cv2.imread("./images/dog.jpeg"), cv2.COLOR_BGR2RGB), (width, width))
# y = torch.tensor(y, dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
# y = y[None, ...]

# for name in fr_names:
#     try:
#         # iqa = pyiqa.create_metric(name.lower(), device=device)
#         iqa = pyiqa.create_metric(name.lower(), device=device, as_loss=True)
#
#         score = iqa(x, y)
#         print(name, score)
#         # score = iqa('./ResultsCalibra/dist_dir/I03.bmp', './ResultsCalibra/ref_dir/I03.bmp')
#         # print(name, score)
#         #
#         # # For FID metric, use directory or precomputed statistics as inputs
#         # # refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
#         # fid_metric = pyiqa.create_metric('fid')
#         # score = fid_metric('./ResultsCalibra/dist_dir/', './ResultsCalibra/ref_dir')
#         # score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")
#     except:
#         print(name, "fail")

x = cv2.resize(cv2.cvtColor(cv2.imread("./images/orange1.jpg"), cv2.COLOR_BGR2RGB), (width, width))
x = torch.tensor(x, dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
x = x[None, ...]

y = cv2.resize(cv2.cvtColor(cv2.imread("./images/orange2.jpg"), cv2.COLOR_BGR2RGB), (width, width))
y = torch.tensor(y, dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
y = y[None, ...]


for name in nr_names:
    try:
        iqa = pyiqa.create_metric(name.lower(), device=device)
        score = [iqa(i) for i in [x, y]]
        print(name, iqa.lower_better, score)
    except:
        print(name, "fail")
