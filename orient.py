import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import argparse

def load_ply(filepath):
    """Load Gaussian splat PLY file."""
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    # Extract positions
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    
    # Extract all other properties
    properties = {}
    for prop in vertex.properties:
        if prop.name not in ['x', 'y', 'z']:
            properties[prop.name] = vertex[prop.name]
    
    return positions, properties, plydata

def estimate_ground_plane_pca(positions, sample_ratio=0.1):
    """
    Estimate ground plane using PCA on bottom percentile of points.
    Assumes ground points are among the lowest Z values.
    """
    # Sample points from lower portion of the scene
    z_threshold = np.percentile(positions[:, 2], 20)
    ground_candidates = positions[positions[:, 2] < z_threshold]
    
    # Subsample for efficiency
    n_samples = int(len(ground_candidates) * sample_ratio)
    if n_samples > 1000:
        indices = np.random.choice(len(ground_candidates), min(n_samples, len(ground_candidates)), replace=False)
        ground_candidates = ground_candidates[indices]
    
    # PCA to find plane normal (smallest variance direction)
    pca = PCA(n_components=3)
    pca.fit(ground_candidates)
    
    # Normal is the direction with smallest variance (last principal component)
    normal = pca.components_[2]
    
    # Ensure normal points upward (positive Z component)
    if normal[2] < 0:
        normal = -normal
    
    centroid = np.mean(ground_candidates, axis=0)
    
    return normal, centroid

def estimate_ground_plane_ransac(positions, iterations=1000, threshold=0.1, sample_ratio=0.2):
    """
    Estimate ground plane using RANSAC on bottom points.
    More robust to outliers than PCA.
    """
    # Focus on lower portion of scene
    z_threshold = np.percentile(positions[:, 2], 30)
    candidates = positions[positions[:, 2] < z_threshold]
    
    # Subsample for efficiency
    n_samples = int(len(candidates) * sample_ratio)
    if n_samples > 5000:
        indices = np.random.choice(len(candidates), n_samples, replace=False)
        candidates = candidates[indices]
    
    best_inliers = 0
    best_normal = None
    best_point = None
    
    for _ in range(iterations):
        # Randomly sample 3 points
        sample_idx = np.random.choice(len(candidates), 3, replace=False)
        sample_points = candidates[sample_idx]
        
        # Compute plane from 3 points
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        
        if np.linalg.norm(normal) < 1e-6:
            continue
        
        normal = normal / np.linalg.norm(normal)
        
        # Count inliers
        distances = np.abs(np.dot(candidates - sample_points[0], normal))
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_point = sample_points[0]
    
    # Ensure normal points upward
    if best_normal[2] < 0:
        best_normal = -best_normal
    
    return best_normal, best_point

def compute_alignment_transform(normal, target_normal=np.array([0, 0, 1])):
    """
    Compute rotation matrix to align source normal with target normal.
    """
    # Normalize vectors
    normal = normal / np.linalg.norm(normal)
    target_normal = target_normal / np.linalg.norm(target_normal)
    
    # Compute rotation axis and angle
    rotation_axis = np.cross(normal, target_normal)
    axis_length = np.linalg.norm(rotation_axis)
    
    if axis_length < 1e-6:
        # Vectors are already aligned or opposite
        if np.dot(normal, target_normal) > 0:
            return np.eye(3)
        else:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            rotation_axis = np.cross(normal, perp)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            return R.from_rotvec(np.pi * rotation_axis).as_matrix()
    
    rotation_axis = rotation_axis / axis_length
    angle = np.arcsin(axis_length)
    
    # Create rotation matrix
    rot_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()
    
    return rot_matrix

def transform_quaternions(quaternions, rotation_matrix):
    """
    Transform quaternions by a rotation matrix.
    Quaternions are in (w, x, y, z) or (x, y, z, w) format.
    """
    rot = R.from_matrix(rotation_matrix)
    
    # Handle both quaternion formats
    if quaternions.shape[1] == 4:
        # Try wxyz format first (common in Gaussian splatting)
        try:
            original_rots = R.from_quat(np.roll(quaternions, -1, axis=1))  # wxyz -> xyzw
            transformed_rots = rot * original_rots
            new_quats = transformed_rots.as_quat()
            return np.roll(new_quats, 1, axis=1)  # xyzw -> wxyz
        except:
            # If that fails, try xyzw format
            original_rots = R.from_quat(quaternions)
            transformed_rots = rot * original_rots
            return transformed_rots.as_quat()
    
    return quaternions

def align_gaussian_splats(input_path, output_path, method='ransac'):
    """
    Main function to align Gaussian splats to ground plane.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path to output PLY file
        method: 'ransac' or 'pca' for plane estimation
    """
    print(f"Loading Gaussian splats from {input_path}...")
    positions, properties, plydata = load_ply(input_path)
    
    print(f"Loaded {len(positions)} splats")
    
    # Estimate ground plane
    print(f"Estimating ground plane using {method.upper()}...")
    if method == 'ransac':
        normal, centroid = estimate_ground_plane_ransac(positions)
    else:
        normal, centroid = estimate_ground_plane_pca(positions)
    
    print(f"Ground plane normal: {normal}")
    print(f"Ground plane centroid: {centroid}")
    
    # Compute alignment transformation
    rot_matrix = compute_alignment_transform(normal)
    
    # Transform positions
    print("Transforming positions...")
    centered_positions = positions - centroid
    aligned_positions = (rot_matrix @ centered_positions.T).T
    aligned_positions += centroid
    
    # Transform quaternions if present
    quat_keys = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    if all(key in properties for key in quat_keys):
        print("Transforming rotation quaternions...")
        quaternions = np.stack([properties[key] for key in quat_keys], axis=1)
        new_quaternions = transform_quaternions(quaternions, rot_matrix)
        for i, key in enumerate(quat_keys):
            properties[key] = new_quaternions[:, i]
    
    # Create new PLY data
    print(f"Saving aligned splats to {output_path}...")
    vertex_data = [
        (aligned_positions[i, 0], aligned_positions[i, 1], aligned_positions[i, 2],
         *[properties[prop.name][i] for prop in plydata['vertex'].properties if prop.name not in ['x', 'y', 'z']])
        for i in range(len(aligned_positions))
    ]
    
    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=plydata['vertex'].data.dtype),
        'vertex'
    )
    
    PlyData([vertex_element]).write(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align Gaussian splats to ground plane')
    parser.add_argument('input', help='Input PLY file path')
    parser.add_argument('output', help='Output PLY file path')
    parser.add_argument('--method', choices=['ransac', 'pca'], default='ransac',
                        help='Method for ground plane estimation (default: ransac)')
    
    args = parser.parse_args()
    
    align_gaussian_splats(args.input, args.output, args.method)