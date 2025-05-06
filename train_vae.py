import os
import shutil
import time
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from thop import profile
from PIL import Image

import Helpers as hf
from vgg19 import VGG19
from RES_VAE_Dynamic import VAE

# ---------- Utility Functions ----------

def compute_gradient_variance(model):
    total_variance = 0.0
    total_params = 0
    for param in model.parameters():
        if param.grad is not None:
            var = torch.var(param.grad.detach()).item()
            total_variance += var
            total_params += 1
    return total_variance / total_params if total_params > 0 else 0.0

def compute_vae_complexity(model, input_size, device):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\n[MACs & Params]\nVAE Model MACs: {macs:,}\nVAE Model Parameters: {params:,}")
    with open("vae_macs_params.txt", "w") as f:
        f.write(f"MACs: {macs:,}\nParameters: {params:,}\n")

def measure_vae_inference_time(model, input_size, device, repeats=100):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    avg_time = (time.time() - start_time) / repeats
    print(f"\n[Inference Timing]\nAverage VAE inference time: {avg_time:.6f} seconds")
    with open("vae_inference_time.txt", "w") as f:
        f.write(f"Average inference time: {avg_time:.6f} seconds\n")

def profile_inference_postprocessing(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        start_inf = time.time()
        recon_img, _, _ = model(image_tensor)
        end_inf = time.time()

        start_post = time.time()
        recon_img = recon_img.cpu().clamp(0, 1)
        _ = transforms.ToPILImage()(recon_img.squeeze(0))
        end_post = time.time()

    return end_inf - start_inf, end_post - start_post

def measure_full_inference_pipeline(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    total_start = time.time()
    start_pre = time.time()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    end_pre = time.time()
    model.eval()
    with torch.no_grad():
        start_inf = time.time()
        output = model(img_tensor)
        end_inf = time.time()
    start_post = time.time()
    recon = output[0].cpu().clamp(0, 1)
    end_post = time.time()
    total_end = time.time()
    print("\n[Pipeline Timing]")
    print(f"Preprocessing time:   {(end_pre - start_pre):.6f} s")
    print(f"Inference time:       {(end_inf - start_inf):.6f} s")
    print(f"Postprocessing time:  {(end_post - start_post):.6f} s")
    print(f"Total processing time:{(total_end - total_start):.6f} s")
    return img_tensor

# ---------- Argument Parsing ----------

parser = argparse.ArgumentParser(description="Training Params")
parser.add_argument("--model_name", "-mn", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", type=str, required=True)
parser.add_argument("--save_dir", "-sd", type=str, default=".")
parser.add_argument("--norm_type", "-nt", type=str, default="bn")
parser.add_argument("--nepoch", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", type=int, default=128)
parser.add_argument("--image_size", "-ims", type=int, default=64)
parser.add_argument("--ch_multi", "-w", type=int, default=64)
parser.add_argument("--num_res_blocks", "-nrb", type=int, default=1)
parser.add_argument("--device_index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", type=int, default=251)
parser.add_argument("--save_interval", "-si", type=int, default=251)
parser.add_argument("--block_widths", "-bw", type=int, nargs='+', default=(1, 2, 4, 8))
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--feature_scale", "-fs", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", type=float, default=1)
parser.add_argument("--load_checkpoint", "-cp", action="store_true")
parser.add_argument("--deep_model", "-dm", action="store_true")
args = parser.parse_args()

# ---------- Device Setup ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}\n")

# ---------- Data Loading ----------
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)
train_size = int(0.9 * len(data_set))
test_size = len(data_set) - train_size
train_set, test_set = torch.utils.data.random_split(data_set, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

dataiter = iter(test_loader)
test_images, _ = next(dataiter)
test_images = test_images.to(device)

vae_net = VAE(
    channel_in=test_images.shape[1],
    ch=args.ch_multi,
    blocks=args.block_widths,
    latent_channels=args.latent_channels,
    num_res_blocks=args.num_res_blocks,
    norm_type=args.norm_type,
    deep_model=args.deep_model
).to(device)

optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

if args.norm_type == "bn":
    print("-Using BatchNorm")
elif args.norm_type == "gn":
    print("-Using GroupNorm")

feature_extractor = VGG19().to(device) if args.feature_scale > 0 else None
print("-VGG19 Feature Loss ON" if feature_extractor else "-VGG19 Feature Loss OFF")

num_model_params = sum(p.numel() for p in vae_net.parameters())
print(f"-This Model Has {num_model_params} (Approximately {num_model_params // 1e6:.0f} Million) Parameters!")
fm_size = args.image_size // (2 ** len(args.block_widths))
print(f"-The Latent Space Size Is {args.latent_channels}x{fm_size}x{fm_size}!")

input_size = (1, 3, args.image_size, args.image_size)
compute_vae_complexity(vae_net, input_size, device)
measure_vae_inference_time(vae_net, input_size, device)

# Full pipeline timing before training
sample_class = os.listdir(args.dataset_root)[0]
sample_img = os.listdir(os.path.join(args.dataset_root, sample_class))[0]
sample_img_path = os.path.join(args.dataset_root, sample_class, sample_img)
print(f"Profiling full pipeline on: {sample_img_path}")
measure_full_inference_pipeline(vae_net, sample_img_path, device)

# ---------- Training Setup ----------
if not os.path.exists(f"{args.save_dir}/Models"):
    os.makedirs(f"{args.save_dir}/Models")
if not os.path.exists(f"{args.save_dir}/Results"):
    os.makedirs(f"{args.save_dir}/Results")

save_file_name = f"{args.model_name}_{args.image_size}"
if args.load_checkpoint:
    checkpoint_path = os.path.join(args.save_dir, "Models", f"{save_file_name}.pt")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("-Checkpoint loaded!")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        vae_net.load_state_dict(checkpoint["model_state_dict"])
        if optimizer.param_groups[0]["lr"] != args.lr:
            print("Updating lr!")
            optimizer.param_groups[0]["lr"] = args.lr
        start_epoch = checkpoint["epoch"]
        data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
    else:
        raise ValueError("Checkpoint does not exist!")
else:
    start_epoch = 0
    data_logger = defaultdict(lambda: [])

# ---------- Timing Tracking ----------
epoch_inference_times = []
epoch_postprocess_times = []

# ---------- Training Loop ----------
for epoch in range(start_epoch, args.nepoch):
    print(f"\rEpoch {epoch + 1}/{args.nepoch}", end="", flush=True)
    vae_net.train()
    epoch_start_time = time.time()
    for i, (images, _) in enumerate(tqdm(train_loader, leave=False)):
        current_iter = i + epoch * len(train_loader)
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=True):
            recon_img, mu, log_var = vae_net(images)
            kl_loss = hf.kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, images)
            loss = args.kl_scale * kl_loss + mse_loss
            if feature_extractor:
                feat_in = torch.cat((recon_img, images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += args.feature_scale * feature_loss
                data_logger["feature_loss"].append(feature_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        grad_var = compute_gradient_variance(vae_net)
        data_logger["grad_variance"].append(grad_var)
        scaler.step(optimizer)
        scaler.update()

        data_logger["mu"].append(mu.mean().item())
        data_logger["mu_var"].append(mu.var().item())
        data_logger["log_var"].append(log_var.mean().item())
        data_logger["log_var_var"].append(log_var.var().item())
        data_logger["kl_loss"].append(kl_loss.item())
        data_logger["img_mse"].append(mse_loss.item())

    # Epoch-end profiling
    test_input = test_images[:1].to(device)
    inf_time, post_time = profile_inference_postprocessing(vae_net, test_input, device)
    epoch_inference_times.append(inf_time)
    epoch_postprocess_times.append(post_time)

# ---------- Final Timing Report ----------
print("\n[Final Timing Summary]")
print(f"Average forward prop time per epoch: {sum(epoch_inference_times)/len(epoch_inference_times):.6f} s")
print(f"Average postprocessing time per epoch: {sum(epoch_postprocess_times)/len(epoch_postprocess_times):.6f} s")
print(f"Total forward prop time per epoch: {sum(epoch_inference_times):.6f} s")
print(f"Total postprocessing time per epoch: {sum(epoch_postprocess_times):.6f} s")