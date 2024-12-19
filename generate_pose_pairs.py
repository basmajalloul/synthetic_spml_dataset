import os
import torch
import numpy as np
from smplx import SMPL
import json
import open3d as o3d

# Set paths
MODEL_PATH = "models/basicmodel_m_lbs_10_207_0_v1.0.0"  # Path to SMPL model files
EXPORT_PATH = "pose_pairs/"  # Path to save pose pairs
os.makedirs(EXPORT_PATH, exist_ok=True)

# Initialize SMPL model
def initialize_smpl(model_path=MODEL_PATH, gender="neutral"):
    smpl = SMPL(model_path=model_path, gender=gender)
    return smpl

# Generate a random pose
def generate_random_pose(smpl):
    pose = torch.randn(1, 72) * 0.2  # Random pose (scaled for realism)
    shape = torch.randn(1, 10) * 0.03  # Random shape
    output = smpl(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=shape)
    vertices = output.vertices.detach().cpu().numpy()
    return pose, vertices[0]

# Generate a slightly modified pose
def modify_pose(base_pose, noise_level=0.1):
    noise = torch.randn_like(base_pose) * noise_level
    modified_pose = base_pose + noise
    return modified_pose

# Create a pair of poses with the specified similarity level
# Create a pair of poses with the specified similarity level
def create_pose_pair(smpl, similarity_level):
    base_pose, _ = generate_random_pose(smpl)

    if similarity_level == 100:  # Identical
        modified_pose = base_pose.clone()
        similarity = "100%"
    elif similarity_level == 75:  # Slightly different
        modified_pose = modify_pose(base_pose, noise_level=0.05)
        similarity = "75%"
    elif similarity_level == 50:  # Moderately different
        modified_pose = modify_pose(base_pose, noise_level=0.1)
        similarity = "50%"
    elif similarity_level == 25:  # Blatantly different
        modified_pose, _ = generate_random_pose(smpl)
        similarity = "25%"

    # Return pose parameter vectors instead of vertices
    return base_pose, modified_pose, similarity


# Save pose pairs to JSON
def save_pose_pair(base_pose, modified_pose, similarity, pair_id):
    # Create subfolder based on similarity level
    similarity_folder = os.path.join(EXPORT_PATH, f"similarity_{similarity}")
    os.makedirs(similarity_folder, exist_ok=True)

    # Save each pose separately
    base_filename = os.path.join(similarity_folder, f"pose_pair_{pair_id}_left.json")
    modified_filename = os.path.join(similarity_folder, f"pose_pair_{pair_id}_right.json")
    similarity_filename = os.path.join(similarity_folder, f"pose_pair_{pair_id}_similarity.json")

    base_data = {"pose": base_pose.tolist()}
    modified_data = {"pose": modified_pose.tolist()}
    similarity_data = {"similarity": similarity}

    with open(base_filename, "w") as f:
        json.dump(base_data, f)
    with open(modified_filename, "w") as f:
        json.dump(modified_data, f)
    with open(similarity_filename, "w") as f:
        json.dump(similarity_data, f)

    print(f"Saved pose pair {pair_id} in {similarity_folder}.")

# Visualize a single pose using Open3D
def visualize_pose(vertices, faces, color, output_path=None):
    # Ensure vertices is a NumPy array
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    # Validate and reshape if needed
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}. Expected (n, 3).")

    # Create the Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    # Visualization and saving
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    if output_path:
        vis.capture_screen_image(output_path)
        print(f"Pose image saved to {output_path}")

    vis.destroy_window()



if __name__ == "__main__":
    smpl = initialize_smpl()

    # Generate pose pairs
    for i in range(500):  # Generate 500 pairs
        similarity_level = np.random.choice([100, 75, 50, 25])  # Random similarity level
        base_pose, modified_pose, similarity = create_pose_pair(smpl, similarity_level)
        save_pose_pair(base_pose, modified_pose, similarity, pair_id=i + 1)

        base_pose = base_pose.flatten()
        modified_pose = modified_pose.flatten()

        # Generate vertices for visualization
        base_output = smpl(
            global_orient=torch.tensor(base_pose[:3].reshape(1, 3), dtype=torch.float32),
            body_pose=torch.tensor(base_pose[3:].reshape(1, 69), dtype=torch.float32),
        )
        base_vertices = base_output.vertices.detach().cpu().numpy()[0]

        modified_output = smpl(
            global_orient=torch.tensor(modified_pose[:3].reshape(1, 3), dtype=torch.float32),
            body_pose=torch.tensor(modified_pose[3:].reshape(1, 69), dtype=torch.float32),
        )
        modified_vertices = modified_output.vertices.detach().cpu().numpy()[0]

        os.makedirs("pose_pairs_images", exist_ok=True)

        # Visualize the poses
        visualize_pose(
            base_vertices, smpl.faces, [1, 0, 0], output_path=f"pose_pairs_images/base_pose_{i + 1}.png"
        )
        visualize_pose(
            modified_vertices, smpl.faces, [0, 1, 0], output_path=f"pose_pairs_images/modified_pose_{i + 1}.png"
        )


