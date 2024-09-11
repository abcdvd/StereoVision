from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

target = listdir('/Users/imajne/Desktop/Python_Project/StereoVIsion/data')
target.sort()
target.pop(0)

def SGBM(name):
    left_img = cv2.imread(f'data/{name}/im0.png', 0)
    right_img = cv2.imread(f'data/{name}/im1.png', 0)

    text = open(f"./data/{name}/calib.txt", 'r')
    cam0 = text.readline()
    focal_length = float(cam0[cam0.find('[')+1 : cam0.find(' ')])
    text.readline()
    text.readline()

    baseline = float(text.readline()[9:])
    text.readline()
    text.readline()
    ndisp = int(text.readline()[6:])
    vmin = int(text.readline()[5:])
    vmax = int(text.readline()[5:])
    text.close()

    # Create StereoSGBM object using the parameters
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=vmin,  # vmin
        numDisparities=ndisp,  # ndisp
        blockSize=10,  # Typical value
        P1=8 * 3 * 5 ** 2,  # Smoothness parameter 1
        P2=32 * 3 * 5 ** 2,  # Smoothness parameter 2
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the disparity map
    disparity_sgbm = stereo_sgbm.compute(left_img, right_img).astype(np.float32) / 16.0

    # Normalize disparity for visualization
    disparity_sgbm_normalized = cv2.normalize(disparity_sgbm, disparity_sgbm, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_sgbm_normalized = np.uint8(disparity_sgbm_normalized)

    # Save the normalized disparity map as a PNG image
    cv2.imwrite(f'./disparity_result/{name}.png', disparity_sgbm_normalized)

    print(f"{name} Disparity image saved successfully!")

    # Constants
    baseline /= 1000  # Baseline in meters (converted from mm to meters)

    # Avoid division by zero in disparity
    disparity_sgbm[disparity_sgbm == 0] = 0.1

    # Compute depth map
    depth_map = (focal_length * baseline) / disparity_sgbm

    # Normalize and visualize depth map
    depth_map_normalized = cv2.normalize(depth_map, depth_map, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    cv2.imwrite(f'./depth_result/{name}.png', depth_map_normalized)

    print(f"{name} Depth image saved successfully!")




for item in target:
    SGBM(item)