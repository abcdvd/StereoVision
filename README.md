# Stereo Disparity and Depth Map Calculation

This Python script calculates the disparity map and depth map from a pair of rectified stereo images using the Semi-Global Block Matching (SGBM) algorithm.

## Overview

The script processes a directory of stereo image pairs. For each pair, it:

1.  Reads the left and right images.
2.  Reads calibration parameters (focal length, baseline, disparity range, etc.) from a `calib.txt` file associated with the image pair.
3.  Computes the disparity map using `cv2.StereoSGBM_create`.
4.  Normalizes and saves the disparity map as a PNG image.
5.  Calculates the depth map using the formula:
    ```
    Depth = (focal\_length * baseline) / disparity
    ```
6.  Normalizes and saves the depth map as a PNG image.

## Prerequisites

-   Python 3
-   OpenCV (`cv2`)
-   NumPy (`numpy`)

You can install these libraries using pip:
```bash
pip install opencv-python numpy
