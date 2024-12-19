# Synthetic Pose Similarity Dataset

## Overview
This repository contains a synthetic dataset designed to validate a Pose Similarity Metric (PSM) for evaluating human poses. The dataset was generated using the Skinned Multi-Person Linear (SMPL) model, providing a controlled environment to test pose similarity under varying conditions.

## Dataset Description
The dataset consists of **500 pose pairs** categorized into four similarity levels:

| Similarity Level (%) | Number of Pairs | Description |
|-----------------------|-----------------|-------------|
| **100%**             | 125             | Identical poses with no variations. |
| **75%**              | 125             | Slightly different poses with small angular variations. |
| **50%**              | 125             | Moderately different poses with clear joint angle changes. |
| **25%**              | 125             | Blatantly different poses with significant variations. |

### File Structure
The dataset is organized into subfolders based on similarity levels:

```
pose_pairs/
    similarity_100/
        pose_pair_1_left.json
        pose_pair_1_right.json
        pose_pair_1_similarity.json
        ...
    similarity_75/
    similarity_50/
    similarity_25/
pose_pairs_images/
    base_pose_1.png
    modified_pose_1.png
    ...
scripts/
    generate_pose_pairs.py
```

Each pose pair includes:
- **Left Pose JSON**: Base pose in SMPL parameter format.
- **Right Pose JSON**: Modified pose in SMPL parameter format.
- **Similarity JSON**: Ground truth similarity level for the pair.

The `pose_pairs_images/` directory contains rendered images of a subset of pose pairs, providing a qualitative view of the generated poses and their variations.

## Data Format
Each pose file (e.g., `pose_pair_1_left.json`) contains:
```json
{
  "pose": [/* 72 pose parameters */],
  "similarity": "100%"
}
```

## Dataset Generation
The dataset was generated using a Python script leveraging the SMPL model for pose synthesis and controlled modifications. The script allows for creating pose pairs with specified similarity levels, introducing variations such as noise and occlusions. 

The script is included in this repository under the `scripts/` directory:
```
scripts/
    generate_pose_pairs.py
```

### Running the Script
To generate the dataset:
1. Install the required dependencies (e.g., SMPL, Open3D, NumPy, PyTorch):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python scripts/generate_pose_pairs.py
   ```

The generated dataset will be saved in the `pose_pairs/` directory.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Load the JSON files into your preferred programming environment for analysis or visualization.

3. Utilize the similarity level for validating Pose Similarity Metrics (PSM).

### Visualization
The dataset includes rendered images of a subset of pose pairs in the `pose_pairs_images/` directory. These images provide a qualitative view of the generated poses.

## Citation
If you use this dataset in your research, please cite the following:

Loper, Matthew, et al. "SMPL: A Skinned Multi-Person Linear Model." *ACM Transactions on Graphics (TOG)*, vol. 34, no. 6, 2015, pp. 248:1–248:16. DOI: [10.1145/2816795.2818013](https://doi.org/10.1145/2816795.2818013).

and my paper:

.... et al. "" 

## License
This dataset is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
This dataset was generated using the SMPL model developed by the Max Planck Institute for Intelligent Systems. For more information, visit the [official SMPL website](https://smpl.is.tue.mpg.de/index.html).
