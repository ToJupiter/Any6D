(Any6D) root@C.27272223:/workspace/Any6D$ python run_demo.py --img_to_3d
/venv/Any6D/lib/python3.9/site-packages/kornia/feature/lightglue.py:44: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
Warp 1.0.2 initialized:
   CUDA Toolkit 11.5, Driver 12.8
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA A10" (22 GiB, sm_86, mempool enabled)
   Kernel cache:
     /root/.cache/warp/1.0.2
[__init__()] self.h5_file:None
/workspace/Any6D/foundationpose/learning/training/predict_score.py:151: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt = torch.load(ckpt_dir)
[__init__()] self.h5_file:
/workspace/Any6D/foundationpose/learning/training/predict_pose_refine.py:138: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt = torch.load(ckpt_dir)
/venv/Any6D/lib/python3.9/site-packages/torch/utils/cpp_extension.py:1965: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
[seed_everything()] Seed set to 0
[_load_checkpoint()] Loaded checkpoint sucessfully
[set_image()] For numpy array image, we assume (HxWxC) format
[set_image()] Computing image embeddings for the provided image...
[set_image()] Image embeddings computed.
Downloading data from 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx' to file '/root/.u2net/u2net.onnx'.
100%|███████████████████████████████████████| 176M/176M [00:00<00:00, 1.64TB/s]

Loading pipeline components...:   0%|                                                                                                           | 0/8 [00:00<?, ?it/s]
Loading pipeline components...:  38%|█████████████████████████████████████▏                                                             | 3/8 [00:00<00:00, 12.90it/s]
Loading pipeline components...:  62%|█████████████████████████████████████████████████████████████▉                                     | 5/8 [00:00<00:00, 12.46it/s]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:01<00:00,  5.22it/s]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:01<00:00,  6.19it/s]
/venv/instantmesh-py310/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()

  0%|                                                                                                                                          | 0/75 [00:00<?, ?it/s]
  1%|█▋                                                                                                                                | 1/75 [00:00<00:28,  2.59it/s]
  4%|█████▏                                                                                                                            | 3/75 [00:00<00:14,  4.92it/s]
  5%|██████▉                                                                                                                           | 4/75 [00:00<00:14,  4.98it/s]
  7%|████████▋                                                                                                                         | 5/75 [00:01<00:13,  5.03it/s]
  8%|██████████▍                                                                                                                       | 6/75 [00:01<00:13,  4.96it/s]
  9%|████████████▏                                                                                                                     | 7/75 [00:01<00:13,  5.01it/s]
 11%|█████████████▊                                                                                                                    | 8/75 [00:01<00:13,  5.04it/s]
 12%|███████████████▌                                                                                                                  | 9/75 [00:01<00:13,  5.06it/s]
 13%|█████████████████▏                                                                                                               | 10/75 [00:02<00:12,  5.07it/s]
 15%|██████████████████▉                                                                                                              | 11/75 [00:02<00:12,  5.04it/s]
 16%|████████████████████▋                                                                                                            | 12/75 [00:02<00:12,  5.05it/s]
 17%|██████████████████████▎                                                                                                          | 13/75 [00:02<00:12,  5.06it/s]
 19%|████████████████████████                                                                                                         | 14/75 [00:02<00:12,  5.07it/s]
 20%|█████████████████████████▊                                                                                                       | 15/75 [00:03<00:11,  5.06it/s]
 21%|███████████████████████████▌                                                                                                     | 16/75 [00:03<00:11,  5.05it/s]
 23%|█████████████████████████████▏                                                                                                   | 17/75 [00:03<00:11,  5.04it/s]
 24%|██████████████████████████████▉                                                                                                  | 18/75 [00:03<00:11,  5.05it/s]
 25%|████████████████████████████████▋                                                                                                | 19/75 [00:03<00:11,  5.06it/s]
 27%|██████████████████████████████████▍                                                                                              | 20/75 [00:04<00:10,  5.06it/s]
 28%|████████████████████████████████████                                                                                             | 21/75 [00:04<00:10,  5.06it/s]
 29%|█████████████████████████████████████▊                                                                                           | 22/75 [00:04<00:10,  5.04it/s]
 31%|███████████████████████████████████████▌                                                                                         | 23/75 [00:04<00:10,  5.06it/s]
 32%|█████████████████████████████████████████▎                                                                                       | 24/75 [00:04<00:10,  5.06it/s]
 33%|███████████████████████████████████████████                                                                                      | 25/75 [00:05<00:09,  5.05it/s]
 35%|████████████████████████████████████████████▋                                                                                    | 26/75 [00:05<00:09,  5.06it/s]
 36%|██████████████████████████████████████████████▍                                                                                  | 27/75 [00:05<00:09,  5.05it/s]
 37%|████████████████████████████████████████████████▏                                                                                | 28/75 [00:05<00:09,  5.05it/s]
 39%|█████████████████████████████████████████████████▉                                                                               | 29/75 [00:05<00:09,  5.06it/s]
 40%|███████████████████████████████████████████████████▌                                                                             | 30/75 [00:06<00:08,  5.05it/s]
 41%|█████████████████████████████████████████████████████▎                                                                           | 31/75 [00:06<00:08,  5.05it/s]
 43%|███████████████████████████████████████████████████████                                                                          | 32/75 [00:06<00:08,  5.05it/s]
 44%|████████████████████████████████████████████████████████▊                                                                        | 33/75 [00:06<00:08,  5.05it/s]
 45%|██████████████████████████████████████████████████████████▍                                                                      | 34/75 [00:06<00:08,  5.05it/s]
 47%|████████████████████████████████████████████████████████████▏                                                                    | 35/75 [00:06<00:07,  5.04it/s]
 48%|█████████████████████████████████████████████████████████████▉                                                                   | 36/75 [00:07<00:07,  5.04it/s]
 49%|███████████████████████████████████████████████████████████████▋                                                                 | 37/75 [00:07<00:07,  5.04it/s]
 51%|█████████████████████████████████████████████████████████████████▎                                                               | 38/75 [00:07<00:07,  5.03it/s]
 52%|███████████████████████████████████████████████████████████████████                                                              | 39/75 [00:07<00:07,  5.04it/s]
 53%|████████████████████████████████████████████████████████████████████▊                                                            | 40/75 [00:07<00:06,  5.03it/s]
 55%|██████████████████████████████████████████████████████████████████████▌                                                          | 41/75 [00:08<00:06,  5.04it/s]
 56%|████████████████████████████████████████████████████████████████████████▏                                                        | 42/75 [00:08<00:06,  5.04it/s]
 57%|█████████████████████████████████████████████████████████████████████████▉                                                       | 43/75 [00:08<00:06,  5.05it/s]
 59%|███████████████████████████████████████████████████████████████████████████▋                                                     | 44/75 [00:08<00:06,  5.05it/s]
 60%|█████████████████████████████████████████████████████████████████████████████▍                                                   | 45/75 [00:08<00:05,  5.03it/s]
 61%|███████████████████████████████████████████████████████████████████████████████                                                  | 46/75 [00:09<00:05,  5.04it/s]
 63%|████████████████████████████████████████████████████████████████████████████████▊                                                | 47/75 [00:09<00:05,  5.04it/s]
 64%|██████████████████████████████████████████████████████████████████████████████████▌                                              | 48/75 [00:09<00:05,  5.03it/s]
 65%|████████████████████████████████████████████████████████████████████████████████████▎                                            | 49/75 [00:09<00:05,  5.03it/s]
 67%|██████████████████████████████████████████████████████████████████████████████████████                                           | 50/75 [00:09<00:04,  5.01it/s]
 68%|███████████████████████████████████████████████████████████████████████████████████████▋                                         | 51/75 [00:10<00:04,  5.01it/s]
 69%|█████████████████████████████████████████████████████████████████████████████████████████▍                                       | 52/75 [00:10<00:04,  5.03it/s]
 71%|███████████████████████████████████████████████████████████████████████████████████████████▏                                     | 53/75 [00:10<00:04,  5.03it/s]
 72%|████████████████████████████████████████████████████████████████████████████████████████████▉                                    | 54/75 [00:10<00:04,  5.02it/s]
 73%|██████████████████████████████████████████████████████████████████████████████████████████████▌                                  | 55/75 [00:10<00:03,  5.03it/s]
 75%|████████████████████████████████████████████████████████████████████████████████████████████████▎                                | 56/75 [00:11<00:03,  5.02it/s]
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████                               | 57/75 [00:11<00:03,  5.03it/s]
 77%|███████████████████████████████████████████████████████████████████████████████████████████████████▊                             | 58/75 [00:11<00:03,  5.03it/s]
 79%|█████████████████████████████████████████████████████████████████████████████████████████████████████▍                           | 59/75 [00:11<00:03,  5.04it/s]
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████▏                         | 60/75 [00:11<00:02,  5.04it/s]
 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 61/75 [00:12<00:02,  5.04it/s]
 83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                      | 62/75 [00:12<00:02,  5.04it/s]
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                    | 63/75 [00:12<00:02,  5.05it/s]
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████                   | 64/75 [00:12<00:02,  5.04it/s]
 87%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                 | 65/75 [00:12<00:01,  5.03it/s]
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌               | 66/75 [00:13<00:01,  5.03it/s]
 89%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏             | 67/75 [00:13<00:01,  5.01it/s]
 91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉            | 68/75 [00:13<00:01,  5.03it/s]
 92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋          | 69/75 [00:13<00:01,  5.03it/s]
 93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍        | 70/75 [00:13<00:00,  5.03it/s]
 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████       | 71/75 [00:14<00:00,  5.03it/s]
 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊     | 72/75 [00:14<00:00,  5.01it/s]
 97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌   | 73/75 [00:14<00:00,  5.02it/s]
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎ | 74/75 [00:14<00:00,  5.03it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [00:14<00:00,  5.03it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [00:14<00:00,  5.02it/s]
Traceback (most recent call last):
  File "/workspace/Any6D/tools/instantmesh_worker.py", line 106, in <module>
    main()
  File "/workspace/Any6D/tools/instantmesh_worker.py", line 102, in main
    instant_mesh_process(images, args.config_path, args.output_mesh)
  File "/workspace/Any6D/tools/instantmesh_worker.py", line 59, in instant_mesh_process
    from instantmesh.src.utils.train_util import instantiate_from_config
ModuleNotFoundError: No module named 'instantmesh'

ERROR conda.cli.main_run:execute(125): `conda run python tools/instantmesh_worker.py --input_image results/demo_mustard/input_demo_mustard.png --output_mesh results/demo_mustard/mesh_demo_mustard.obj --name demo_mustard --config_path ./instantmesh/configs/instant-mesh-large.yaml --save_dir results/demo_mustard` failed. (See above for error)
Traceback (most recent call last):
  File "/workspace/Any6D/run_demo.py", line 76, in <module>
    subprocess.run(cmd, check=True)
  File "/venv/Any6D/lib/python3.9/subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['conda', 'run', '-n', 'instantmesh-py310', 'python', 'tools/instantmesh_worker.py', '--input_image', 'results/demo_mustard/input_demo_mustard.png', '--output_mesh', 'results/demo_mustard/mesh_demo_mustard.obj', '--name', 'demo_mustard', '--config_path', './instantmesh/configs/instant-mesh-large.yaml', '--save_dir', 'results/demo_mustard']' returned non-zero exit status 1.