import cv2
import numpy as np

def estimate_motion(img1, img2, K):
    # Step 1: Detect and match features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 2: Estimate the fundamental matrix
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Step 3: Estimate the essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    return R, t

# Load your images
img1 = cv2.imread('./temp/img/rgb_b/4.png', 0)
img2 = cv2.imread('./temp/img/rgb_b/5.png', 0)

# Define your camera matrix K
# 计算相机内参
width = 1280
height = 720
fieldOfView = 1.57  # 视场角，单位：弧度
# 计算焦距
fx = fy = 0.5 * width / np.tan(0.5 * fieldOfView)
# 计算主点
cx = width / 2
cy = height / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Estimate the motion
R, t = estimate_motion(img1, img2, K)

# Calculate the position of the second camera in the world coordinate system
t_world = -np.linalg.inv(R) @ t

# Calculate the orientation of the second camera in the world coordinate system
R_world = R.T

print("Position: ", t_world)
print("Orientation: ", R_world)
