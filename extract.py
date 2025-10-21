import os
import uuid
import sys
import torch
import numpy as np
import open3d as o3d
from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


# === CONFIG ===
model_dir = "/home/te/projects/splat_rigid_body/data/gaussianGrouping/output"
iteration = 30000
target_label = 0   # <-- the label (or mask ID) of the object you want
num_classes = 4    # number of classes from your config

# === LOAD MODEL ===
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
# Add an argument for the configuration file
parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)

args.source_path = "/home/te/projects/splat_rigid_body/data/gaussianGrouping"

dataset = lp.extract(args)

prepare_output_and_logger(dataset)
gaussians = GaussianModel(dataset.sh_degree)
print("Gaussian Keys ", gaussians.__dict__.keys())
scene = Scene(dataset, gaussians)

# dataset = ModelParams(model_dir)
# pipe = PipelineParams()
# gaussians = GaussianModel(dataset.sh_degree)
# scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

# === LOAD CLASSIFIER ===
classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
classifier.cuda()
classifier.load_state_dict(
    torch.load(f"{model_dir}/point_cloud/iteration_{iteration}/classifier.pth")
)
classifier.eval()

# === CLASSIFY GAUSSIANS ===
with torch.no_grad():
    logits3d = classifier(gaussians._objects_dc.permute(2,0,1).unsqueeze(0))
    labels3d = torch.argmax(logits3d, dim=1).squeeze().cpu().numpy()

print("Labels found:", np.unique(labels3d))

# === SELECT TARGET OBJECT ===
mask = labels3d == target_label
print(f"Found {mask.sum()} gaussians for class {target_label}")

xyz = gaussians._xyz.detach().cpu().numpy()[mask]
# colors = gaussians._features_dc.T.clamp(0,1).cpu().detach().numpy()[mask]
print("features_dc shape:", gaussians._features_dc.shape)
colors = gaussians._features_dc.squeeze(1).clamp(0, 1).cpu().detach().numpy()[mask]



# === SAVE OR VISUALIZE ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(f"{model_dir}/object_{target_label}.ply", pcd)

print(f"Saved object_{target_label}.ply with {len(xyz)} points")
