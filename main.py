import torch
from RES_VAE_Dynamic import VAE
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import argparse
import os

# 0. Argument parser
parser = argparse.ArgumentParser(description="VAE Inference Script")
parser.add_argument("--model_path",   type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--image_path",   type=str, required=True, help="Image or folder")
parser.add_argument("--output_path",  type=str, default="reconstructed.png", help="Output file/folder")
args = parser.parse_args()

# 1. Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. Build exactly the training‐time VAE
vae = VAE(
    channel_in=3, ch=64,
    blocks=(1,2,4,8),
    latent_channels=251,
    num_res_blocks=1,
    norm_type="bn",
    deep_model=False
).to(device)

# 3. Load checkpoint intelligently
ckpt = torch.load(args.model_path, map_location=device)
state = ckpt["model_state_dict"]

# Try unfused (strict) first
loaded_strict = True
try:
    vae.load_state_dict(state, strict=True)
    print("Loaded unfused checkpoint (strict).")
except RuntimeError:
    loaded_strict = False
    print(" Strict load failed → assuming fused checkpoint.")

if not loaded_strict:
    # For fused: swap every ReparamConvBlock into deploy form…
    vae.convert_to_deploy()
    # …then re‐load strictly (now only reparam_conv.* keys exist)
    vae.load_state_dict(state, strict=True)
    print("Converted to deploy then loaded fused weights (strict).")

# 4. Finalize for inference
vae.convert_to_deploy()
vae.eval()
print("Model is in eval/deploy mode.")

# 5. Preprocessing ([-1,1] regime)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])

def reconstruct_and_save(x, out_path):
    with torch.no_grad():
        recon, _, _ = vae(x)

    # De-normalize from [-1,1] back to [0,1]
    recon = recon * 0.5 + 0.5
    recon = recon.clamp(0, 1)

    # Convert to PIL Image and save—this preserves correct RGB scaling
    pil_img = transforms.ToPILImage()(recon.squeeze(0).cpu())
    pil_img.save(out_file)

# 6. Run
if os.path.isdir(args.image_path):
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Batch inference on {args.image_path}")
    for fname in os.listdir(args.image_path):
        if fname.lower().endswith((".jpg",".jpeg",".png")):
            img = Image.open(os.path.join(args.image_path, fname)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            out_file = os.path.join(
                args.output_path,
                f"reconstructed_{os.path.splitext(fname)[0]}.png"
            )
            reconstruct_and_save(x, out_file)
            print("Saved", out_file)
else:
    print(f"Single image inference on {args.image_path}")
    img = Image.open(args.image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    reconstruct_and_save(x, args.output_path)
    print("Saved", args.output_path)