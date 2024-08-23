import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(
    model,
    image_size,
    prompt,
    uc,
    sampler,
    step=20,
    scale=7.5,
    batch_size=8,
    ddim_eta=0.0,
    dtype=torch.float32,
    device="cuda",
    camera=None,
    num_frames=1,
    masks=None,
):
    if "|" in scale:
        scale = [float(w.strip()) for w in scale.split("|")]
    else:
        scale = [float(scale)]
    if "|" in prompt:
        prompt = [x.strip() for x in prompt.split("|")]
        assert (
            len(prompt) == len(scale) or len(scale) == 1
        ), "scale must be a single value or have the same length as prompt"
        print(f"composing {prompt}...")
        # using equal positive weights (conjunction) for all prompts...
        if len(scale) == 1:
            scale = scale * len(prompt)
    print(f"generating image with prompt: {prompt}")
    print(f"using scale: {scale}")
    scale = torch.tensor(scale * batch_size, device=device).reshape(batch_size, -1, 1, 1, 1)  # [num_prompt, 1, 1, 1]

    if type(prompt) != list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)  # [num_prompt, 77, 1024]
        c_ = {"context": c.repeat(batch_size, 1, 1, 1)}  # [batch_size, num_prompt, 77, 1024]
        uc_ = {"context": uc.repeat(batch_size, len(prompt), 1, 1)}  # [batch_size, 1, 77, 1024]
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
            masks=masks,
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface"
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="load model from local config (override model_name)"
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--scale", type=str, default="10")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    # prepare masks

    # ==== a man wearing hat + shirt ====
    # masks = torch.zeros((3, args.size, args.size), dtype=dtype, device=device)
    # masks[0] = 1
    # masks[1, : args.size // 4, :] = 1
    # masks[2, args.size // 3 : -args.size // 3, :] = 1

    # ==== 4 views ====
    masks = torch.zeros((4, 3, args.size, args.size), dtype=dtype, device=device)
    masks[0, 0] = 1
    masks[0, 1, :, :] = 1
    masks[0, 2, :, -int(args.size / 2) :] = 1

    masks[1] = 1

    masks[2, 0] = 1
    masks[2, 1, :, :] = 1
    masks[2, 2, :, : int(args.size / 2)] = 1

    masks[3] = 1

    # ==== one prompt ====
    # masks = torch.ones((1, args.size, args.size), dtype=dtype, device=device)

    if len(masks.shape) == 3:
        masks = masks.unsqueeze(1).repeat(1, 4, 1, 1)
        masks = F.interpolate(masks, scale_factor=1 / 8, mode="nearest")
    elif len(masks.shape) == 4:
        masks_image = masks.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0 / 1.5
        masks_image = np.concatenate(masks_image, 1).astype(np.uint8)
        masks = F.interpolate(masks, scale_factor=1 / 8, mode="nearest")
        masks = masks.unsqueeze(2).repeat(1, 1, 4, 1, 1)

    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    model.device = device
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning([""]).to(device)
    print("load t2i model done . ")

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(
            args.num_frames,
            elevation=args.camera_elev,
            azimuth_start=args.camera_azim,
            azimuth_span=args.camera_azim_span,
        )
        camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
    else:
        camera = None

    # t = args.text + args.suffix
    t = "|".join([x.strip() + args.suffix for x in args.text.split("|")])
    set_seed(args.seed)
    images = []
    for j in range(4):
        img = t2i(
            model,
            args.size,
            t,
            uc,
            sampler,
            step=50,
            scale=args.scale,
            batch_size=batch_size,
            ddim_eta=0.0,
            dtype=dtype,
            device=device,
            camera=camera,
            num_frames=args.num_frames,
            masks=masks,
        )
        img = np.concatenate(img, 1)
        images.append(img)
    # print(images[0].shape, masks_image.shape)
    images.append(masks_image)
    images = np.concatenate(images, 0)
    os.makedirs("results", exist_ok=True)
    to_save_path = f"results/{args.text}.png"
    Image.fromarray(images).save(to_save_path)
