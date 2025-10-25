# InstantMesh integration via subprocess (Python 3.10)

This document explains how to run the InstantMesh parts of the pipeline in a separate Python 3.10 environment and integrate them back into Any6D with minimal code changes.

## Why isolate InstantMesh?

InstantMesh depends on packages and CUDA/PyTorch versions that conflict with the rest of Any6D. To avoid breakage, we run only the InstantMesh-specific steps in a dedicated Python 3.10 environment and keep the rest of Any6D unchanged.

## What must run in the InstantMesh env

From `sam2_instantmesh.py`, only the following should run inside the InstantMesh (py310) environment:

- `diffusion_image_generation(...)`
  - Dependencies: `torch`, `einops`, `omegaconf`, `diffusers` (`DiffusionPipeline`, `EulerAncestralDiscreteScheduler`), the custom pipeline `instantmesh/zero123plus`, InstantMesh config at `./instantmesh/configs/instant-mesh-large.yaml`, and UNet weights pointed to by `infer_config.unet_path` in that config.
- `instant_mesh_process(images, ...)`
  - Dependencies: `torch`, `torchvision`, `omegaconf`, `instantmesh.src.utils.train_util.instantiate_from_config`, `instantmesh.src.utils.camera_util.get_zero123plus_input_cameras`, InstantMesh checkpoint pointed to by `infer_config.model_path` in the same config, and mesh extraction utilities.
- Helper used by `instant_mesh_process`: `save_obj(...)` (can be embedded in worker to avoid importing non-InstantMesh pieces).

Everything else (SAM2 box refinement and image preprocessing, including background removal with `rembg`) can remain in the main Any6D environment.

## Minimal design: subprocess worker

Create a small worker script that lives in this repository and runs entirely within the InstantMesh env. The main pipeline will:

1. Prepare input (background-removed, centered, optionally flipped) PNG in the main env as it currently does via `preprocess_image(...)`.
2. Call the worker via a subprocess to perform:
   - Diffusion multi-view image generation (Zero123++)
   - Mesh reconstruction (InstantMesh)
3. Continue the Any6D pipeline with the produced OBJ mesh (align, pose estimation, evaluation).

### Contract (inputs/outputs)

- Inputs to worker:
  - `--input_image`: Path to the preprocessed RGBA PNG (e.g., `results/<obj>/input_<name>.png`)
  - `--output_mesh`: Output path for OBJ (e.g., `results/<obj>/mesh_<name>.obj`)
  - `--config_path` (optional): Defaults to `./instantmesh/configs/instant-mesh-large.yaml`
  - `--save_dir` (optional): Directory for intermediate artifacts (e.g., the 6-view PNG); defaults to the output mesh directory.
- Outputs from worker:
  - OBJ mesh written at `--output_mesh`
  - Optionally `6_views_<name>.png` in `--save_dir`

### Suggested file: `tools/instantmesh_worker.py`

A minimal worker that reuses logic from `sam2_instantmesh.py` without importing `rembg` or `sam2` at import time.

Pseudo-skeleton:

```python
# tools/instantmesh_worker.py
import os, argparse, torch, numpy as np, torchvision
from PIL import Image
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from einops import rearrange
from instantmesh.src.utils.train_util import instantiate_from_config
from instantmesh.src.utils.camera_util import get_zero123plus_input_cameras


def save_obj(vertices, faces, colors, fpath):
    import numpy as np, trimesh
    vertices = vertices @ np.array([[1,0,0],[0,1,0],[0,0,-1]])
    faces = faces[:, [2,1,0]]
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors).export(fpath, 'obj')


def diffusion_image_generation(input_image, config_path, save_dir, name):
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
    # white-background UNet
    state_dict = torch.load(config.infer_config.unet_path, map_location='cpu', weights_only=True)
    pipe.unet.load_state_dict(state_dict, strict=True)
    pipe = pipe.to(device)

    image = Image.open(input_image).convert('RGBA')
    out = pipe(image, num_inference_steps=75).images[0]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out.save(os.path.join(save_dir, f'6_views_{name}.png'))

    images = np.asarray(out, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2,0,1).contiguous().float()
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    del pipe
    torch.cuda.empty_cache()
    return images


def instant_mesh_process(images, config_path, output_mesh):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model_config).to(device)
    sd = torch.load(config.infer_config.model_path, weights_only=True)['state_dict']
    sd = {k[14:]: v for k, v in sd.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(sd, strict=True)
    model.init_flexicubes_geometry(device, fovy=30.0)
    model.eval()

    images = images.unsqueeze(0).to(device)
    images = torchvision.transforms.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0,1)

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
    ap.add_argument('--input_image', required=True)
    ap.add_argument('--output_mesh', required=True)
    ap.add_argument('--name', default='obj')
    ap.add_argument('--config_path', default='./instantmesh/configs/instant-mesh-large.yaml')
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    images = diffusion_image_generation(args.input_image, args.config_path, args.save_dir or os.path.dirname(args.output_mesh), args.name)
    instant_mesh_process(images, args.config_path, args.output_mesh)

if __name__ == '__main__':
    main()
```

> Note: The worker intentionally avoids importing `sam2_instantmesh.py` to prevent pulling in `rembg` and `sam2` into the InstantMesh env.

## Minimal change to `run_demo.py`

Replace the two direct calls to InstantMesh functions with a subprocess call to the worker. Keep `running_sam_box(...)` and `preprocess_image(...)` in the main env as-is.

Current:

```python
input_image = preprocess_image(color, mask_refine, save_path, obj)
images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
instant_mesh_process(images, save_path, obj)
```

Change to:

```python
import subprocess, sys, shutil

# Ensure the input image is saved by preprocess_image
input_image = preprocess_image(color, mask_refine, save_path, obj)
input_png = os.path.join(save_path, f'input_{obj}.png')  # already saved by preprocess_image
output_mesh = os.path.join(save_path, f'mesh_{obj}.obj')

# Choose how to launch the worker: (A) conda run, or (B) venv python path
cmd = [
    'conda', 'run', '-n', 'instantmesh-py310',
    'python', 'tools/instantmesh_worker.py',
    '--input_image', input_png,
    '--output_mesh', output_mesh,
    '--name', obj,
    '--config_path', './instantmesh/configs/instant-mesh-large.yaml',
    '--save_dir', save_path,
]
subprocess.run(cmd, check=True)

# Continue with the existing steps
mesh = trimesh.load(output_mesh)
mesh = align_mesh_to_coordinate(mesh)
mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))
mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
```

- If not using conda, replace `['conda','run','-n','instantmesh-py310','python']` with the absolute path to the Python of your InstantMesh venv.
- Keep working directory at the repo root so the worker can resolve `./instantmesh/...` paths.

## Environment setup (InstantMesh env)

- Python 3.10 environment (example with conda):

```bash
conda create -n instantmesh-py310 python=3.10 -y
conda activate instantmesh-py310
pip install -r instantmesh/requirements.txt
# Plus required libs used in the worker if missing
pip install diffusers accelerate einops omegaconf torchvision trimesh
# Optional: install instantmesh package path for import stability
pip install -e .
```

- Ensure CUDA is available and that `torch` and `torchvision` versions match your CUDA runtime.
- Ensure the InstantMesh config points to valid weights:
  - `infer_config.unet_path`: custom white-background UNet weights
  - `infer_config.model_path`: InstantMesh model checkpoint

## GPU and caching notes

- The worker uses CUDA if available; otherwise CPU (very slow). Make sure `CUDA_VISIBLE_DEVICES` is configured if you want to pin a GPU.
- Set `HF_HOME` or `TRANSFORMERS_CACHE` to control Hugging Face model cache location and avoid repeated downloads.

## Error handling

- After `subprocess.run(...)`, validate that `output_mesh` exists. If missing, print the worker stderr and abort gracefully.
- You can propagate `--num_inference_steps` and other tunables to the worker later if needed, but keep the initial integration minimal.

## Summary of changes

- Add `tools/instantmesh_worker.py` (new): self-contained script running diffusion + InstantMesh.
- Modify `run_demo.py` (minimal): replace two function calls with a subprocess call to the worker.
- Do not change `sam2_instantmesh.py` (optional future refactor to split imports is possible, but not required for this integration).

## Acceptance checklist

- [ ] `preprocess_image(...)` still runs in the main env and saves `input_<name>.png`.
- [ ] Subprocess runs in the InstantMesh env and produces `mesh_<name>.obj`.
- [ ] `run_demo.py` continues with mesh alignment and Any6D pose pipeline unchanged.
- [ ] No SAM2 or rembg dependencies are required in the InstantMesh env.
