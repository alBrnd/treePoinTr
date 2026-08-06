# -*- coding: utf-8 -*-
"""
Generate complete and partial point clouds from fused ALS/MLS data.


Workflow
--------
1. Read fused ALS/MLS LAS/LAZ point cloud.
2. Downsample complete and MLS point clouds.
3. Normalize both point clouds.
4. Remove statistical outliers from MLS.
5. Compute alpha-shapes in R.
6. Sample complete and partial point clouds.
7. Transform back to original coordinates.
8. Save outputs (.npy and .xyz).

"""

import os
import random
import numpy as np
import laspy
import open3d as o3d
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

import functions

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
R_HOME = "C:/Program Files/R/R-4.3.3"

INPUT_DIR = "./singletrees_clean_sub004/"
GT_OUT = "./singletrees_ALS+MLS_ashapes_v2/"
PARTIAL_OUT = "./singletrees_MLS_ashapes_v2/"
MID_FULL = "./singletrees_mid_fullplys/"
MID_PARTIAL = "./singletrees_mid_partialplys/"
PARTIAL_XYZ = "./singletrees_MLS_ashapes_v2_xyz/"

ALPHA = 0.3
OUTPUT_POINTS = 8192
MAX_POINTS_ALPHA = 75000
SOR_NEIGHBORS = 10
SOR_STD_RATIO = 2.0

os.environ["R_HOME"] = R_HOME

importr("utils")
robjects.r("""
library(alphashape3d)
library(Morpho)
""")

def ensure_directories(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def read_laz_to_numpy(path):
    """Read LAS/LAZ into Nx4 array [x,y,z,platform]."""
    with laspy.open(path) as f:
        las = f.read()
    return np.vstack((las.x, las.y, las.z, las.platform)).T

def normalize_point_cloud(points):
    """Normalize to unit sphere."""
    centroid = np.mean(points, axis=0)
    norm = points - centroid
    scale = np.max(np.linalg.norm(norm, axis=1))
    return norm / scale, centroid, scale

def save_xyz(points, filename):
    np.savetxt(filename, points, fmt="%.6f")

ensure_directories(GT_OUT, PARTIAL_OUT, MID_FULL, MID_PARTIAL, PARTIAL_XYZ)

existing = set(os.listdir(PARTIAL_OUT))
files = [f for f in os.listdir(INPUT_DIR)
         if f.endswith((".las",".laz"))
         and not any(f.startswith(g[:13]) for g in existing)]

for filename in files:
    print(filename)
    cloud = read_laz_to_numpy(os.path.join(INPUT_DIR, filename))
    stem = os.path.splitext(filename)[0]

    # Complete cloud
    full = functions.downsample_point_cloud(cloud[:,:3], MAX_POINTS_ALPHA)
    full_norm, full_centroid, full_scale = normalize_point_cloud(full)

    # MLS cloud only
    mls = cloud[cloud[:,3]==0][:,:3]
    mls = functions.downsample_point_cloud(mls, MAX_POINTS_ALPHA)
    mls_norm, mls_centroid, mls_scale = normalize_point_cloud(mls)

    # Statistical outlier removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mls_norm)
    inlier,_ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS,
        std_ratio=SOR_STD_RATIO)
    mls_norm = np.asarray(inlier.points)

    try:
        gt_name = stem[:13] + "_gt"
        gt = functions.points_from_Rashape3d(
            full_norm,
            nr_points=OUTPUT_POINTS,
            alpha=ALPHA,
            file_path=os.path.join(MID_FULL, gt_name))
        gt = gt * full_scale + full_centroid
        np.save(os.path.join(GT_OUT, gt_name), gt)

        part_name = stem[:13] + "_mls"
        partial = functions.points_from_Rashape3d(
            mls_norm,
            nr_points=random.randrange(OUTPUT_POINTS//4,
                                       OUTPUT_POINTS*3//4),
            alpha=ALPHA,
            file_path=os.path.join(MID_PARTIAL, part_name))
        partial = partial * mls_scale + mls_centroid
        np.save(os.path.join(PARTIAL_OUT, part_name), partial)
        save_xyz(partial, os.path.join(PARTIAL_XYZ, part_name + ".xyz"))

    except Exception as exc:
        print(f"Error processing {filename}: {exc}")

print("Done.")
