import os
import argparse
import torch
import numpy as np
import torchvision
from PIL import Image
from omegaconf import OmegaConf


def save_obj(vertices, faces, colors, fpath):
    import numpy as np
    import trimesh

    vertices = vertices @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    faces = faces[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.export(fpath, 'obj')


def diffusion_image_generation(input_image_path, config_path, save_dir, name):
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    from einops import rearrange

    config = OmegaConf.load(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipe = DiffusionPipeline.from_pretrained(
        'sudo-ai/zero123plus-v1.2',
        custom_pipeline='instantmesh/zero123plus',
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing='trailing'
    )

    # load custom white-background UNet
    state_dict = torch.load(config.infer_config.unet_path, map_location='cpu', weights_only=True)
    pipe.unet.load_state_dict(state_dict, strict=True)
    pipe = pipe.to(device)

    image = Image.open(input_image_path).convert('RGBA')
    out = pipe(image, num_inference_steps=75).images[0]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out.save(os.path.join(save_dir, f'6_views_{name}.png'))

    images = np.asarray(out, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)

    del pipe
    torch.cuda.empty_cache()
    return images


def instant_mesh_process(images, config_path, output_mesh):
    from instantmesh.src.utils.train_util import instantiate_from_config
    from instantmesh.src.utils.camera_util import get_zero123plus_input_cameras

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(config_path)

    model = instantiate_from_config(config.model_config).to(device)
    state = torch.load(config.infer_config.model_path, weights_only=True)['state_dict']
    state = {k[14:]: v for k, v in state.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state, strict=True)

    model.init_flexicubes_geometry(device, fovy=30.0)
    model.eval()

    images = images.unsqueeze(0).to(device)
    images = torchvision.transforms.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    cams = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    with torch.no_grad():
        planes = model.forward_planes(images, cams)
        vertices, faces, colors = model.extract_mesh(planes, use_texture_map=False, **config.infer_config)
        os.makedirs(os.path.dirname(output_mesh), exist_ok=True)
        save_obj(vertices, faces, colors, output_mesh)

    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_image', required=True, help='Path to preprocessed RGBA PNG')
    ap.add_argument('--output_mesh', required=True, help='Path to write output OBJ')
    ap.add_argument('--name', default='obj', help='Name tag for intermediate artifacts')
    ap.add_argument('--config_path', default='./instantmesh/configs/instant-mesh-large.yaml')
    ap.add_argument('--save_dir', default=None, help='Dir to save intermediate outputs (e.g., 6_views_*.png)')
    args = ap.parse_args()

    images = diffusion_image_generation(
        args.input_image,
        args.config_path,
        args.save_dir or os.path.dirname(args.output_mesh),
        args.name,
    )
    instant_mesh_process(images, args.config_path, args.output_mesh)


if __name__ == '__main__':
    main()
