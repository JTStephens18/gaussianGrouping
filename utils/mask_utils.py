import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

input_dir = "/home/te/projects/splat_rigid_body/data/masks/instance_ids_npy"
output_dir = "/home/te/projects/splat_rigid_body/data/gaussianGrouping/object_mask"

os.makedirs(output_dir, exist_ok=True)

tmp = np.load("/home/te/projects/splat_rigid_body/data/masks/instance_ids_npy/00000_instance_id.npy")
print("tmp shape ", tmp.shape)
print("Unique vals ", np.unique(tmp))

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".npy"):
        continue
    mask = np.load(os.path.join(input_dir, fname))
    mask_img = Image.fromarray(mask.astype(np.uint8))
    png_name = fname.replace("_instance_id", "").replace(".npy", ".png")
    mask_img.save(os.path.join(output_dir, png_name))

    # vis = (mask.astype(np.float32) / mask.max()) * 255
    # Image.fromarray(vis.astype(np.uint8)).save(os.path.join(output_dir, "visualization", png_name))