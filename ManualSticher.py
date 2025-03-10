import cv2
import numpy as np
import os

def load_images(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):  # Ensure correct order
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            images.append(img)
    return images

def match_features(img1, img2):
    orb = cv2.ORB_create(nfeatures=5000)  # Use ORB for speed (or use SIFT for accuracy)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detecting keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    #Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    cv2.imshow("Feature Matches", match_img)
    cv2.waitKey(500)

    return kp1, kp2, matches

def stitch_images(img1, img2):
    kp1, kp2, matches = match_features(img1, img2)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        height, width, _ = img2.shape
        img1_warped = cv2.warpPerspective(img1, H, (width * 2, height))

        img1_warped[0:height, 0:width] = img2

        return img1_warped
    else:
        print("Not enough matches found!")
        return None

def stitch_all_images(image_folder):
    images = load_images(image_folder)
    if len(images) < 2:
        print("Need at least two images to stitch.")
        return None

    panorama = images[0]
    for i in range(1, len(images)):
        panorama = stitch_images(panorama, images[i])
        if panorama is None:
            break

    return panorama

image_folder = "Images/"

stitched_result = stitch_all_images(image_folder)

if stitched_result is not None:
    cv2.imshow("Stitched Image", stitched_result)
    cv2.imwrite("stitched_output.jpg", stitched_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
