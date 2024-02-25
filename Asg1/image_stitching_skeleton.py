#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 1
# Name :
# Student ID :
# Email Addr :
#

import cv2
import numpy as np
import argparse


def extract_and_match_feature(img_1, img_2, ratio_test=0.7):
    """
    1/  extract SIFT feature from image 1 and image 2,
    2/  use a bruteforce search to find pairs of matched features:
        for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points

    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_test: ratio for the robustness test
    :return list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    """
    list_pairs_matched_keypoints = []

    # to be completed ....

    # Convert images to grayscale
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # SIFT detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_1, None)
    kp2, des2 = sift.detectAndCompute(gray_2, None)

    # SIFT matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good.append(m)

    for m in good:
        # Extract matching points
        p1 = kp1[m.queryIdx].pt  # Point from img_1
        p2 = kp2[m.trainIdx].pt  # Point from img_2

        # Append as a pair of points [[p1x, p1y], [p2x, p2y]]
        list_pairs_matched_keypoints.append([list(p1), list(p2)])

    # Draw matches on an output image
    img_matches = cv2.drawMatches(img_1, kp1, img_2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the image
    cv2.imwrite('matches.png', img_matches)

    return list_pairs_matched_keypoints


def find_homography_ransac(list_pairs_matched_keypoints,
                           threshold_ratio_inliers=0.85,
                           threshold_reprojection_error=3,
                           max_num_trial=1000):
    """
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points,
    transform the second set of feature point to the first (e.g. warp image 2 to image 1)

    :param list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],...]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples,
                                    accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojection_error: threshold of reprojection error (measured as euclidean distance, in pixels)
                                            to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    """
    best_H = None

    # to be completed ...

    pts_src = np.float32([kp_pair[0] for kp_pair in list_pairs_matched_keypoints])
    pts_dst = np.float32([kp_pair[1] for kp_pair in list_pairs_matched_keypoints])

    best_H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,
                                    ransacReprojThreshold=threshold_reprojection_error,
                                    maxIters=max_num_trial,
                                    confidence=threshold_ratio_inliers)
    
    return best_H

def pyramid_blending(img1, img2, num_levels=6):
    # generate Gaussian pyramid for img1
    G = img1.copy()
    gp1 = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gp1.append(G)

    # generate Gaussian pyramid for img2
    G = img2.copy()
    gp2 = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gp2.append(G)

    # generate Laplacian Pyramid for img1
    lp1 = [gp1[num_levels]]
    for i in range(num_levels, 0, -1):
        GE = cv2.pyrUp(gp1[i])
        L = cv2.subtract(gp1[i-1], GE)
        lp1.append(L)

    # generate Laplacian Pyramid for img2
    lp2 = [gp2[num_levels]]
    for i in range(num_levels, 0, -1):
        GE = cv2.pyrUp(gp2[i])
        L = cv2.subtract(gp2[i-1], GE)
        lp2.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lp1, lp2):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, :cols//2], lb[:, cols//2:]))
        LS.append(ls)

    # now reconstruct
    img_panorama = LS[0]
    for i in range(1, num_levels):
        img_panorama = cv2.pyrUp(img_panorama)
        img_panorama = cv2.add(img_panorama, LS[i])

    return img_panorama

def warp_blend_image(img_1, H, img_2):
    """
    1/  warp image img_2 using the homography H to align it with image img_1
        (using inverse warping and bilinear resampling)
    2/  stitch image img_2 to image img_1 and apply average blending to blend the 2 images into a single panorama image

    :param img_1:  the original first image
    :param H: estimated homography
    :param img_2:the original second image
    :return img_panorama: resulting panorama image
    """
    img_panorama = None

    # Find the inverse of the homography
    H_inv = np.linalg.inv(H)

    # # Get the dimensions of both images
    height_img_1, width_img_1 = img_1.shape[:2]
    height_img_2, width_img_2 = img_2.shape[:2]

    height = img_1.shape[0]+img_2.shape[0] #if want to clip, delete img_2.shape[0]
    width = img_1.shape[1] +img_2.shape[1]

    # Warp the second image with the inverse homography
    dst = cv2.warpPerspective(img_2, H_inv, (width, height))

    for i in range(img_2.shape[0]):
        for j in range(img_2.shape[1]):
            if np.array_equal(dst[i,j],[0,0,0]):
                dst[i,j] = img_1[i,j]
            else:
                dst[i,j][0] = (int(dst[i,j][0]) + int(img_1[i,j][0]))//2  #r
                dst[i,j][1] = (int(dst[i,j][1]) + int(img_1[i,j][1]))//2  #g
                dst[i,j][2] = (int(dst[i,j][2]) + int(img_1[i,j][2]))//2  #b

    img_panorama = dst
    return img_panorama

def stitch_images(img_1, img_2):
    """
    :param img_1: input image 1 is the reference image. We will not warp this image
    :param img_2: We warp this image to align and stich it to the image 1
    :return img_panorama: the resulting stiched image
    """
    print('==================================================================================')
    print('===== stitch two images to generate one panorama image =====')
    print('==================================================================================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_test=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 2 to align it to image 1
    H = find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85,
                               threshold_reprojection_error=3, max_num_trial=1000)

    # ===== warp image 2, blend it with image 1 using average blending to produce the resulting panorama image
    img_panorama = warp_blend_image(img_1=img_1, H=H, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print('==================================================================================')
    print('CSCI3290, Spring 2024, Assignment 1: Image Stitching')
    print('==================================================================================')

    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--im1', type=str, default='test_images/MelakwaLake1.png',
                        help='path of the first input image')
    parser.add_argument('--im2', type=str, default='test_images/MelakwaLake2.png',
                        help='path of the second input image')
    parser.add_argument('--output', type=str, default='MelakwaLake.png',
                        help='the path of the output image')
    args = parser.parse_args()

    # ===== read 2 input images
    img_1 = cv2.imread(args.im1)
    img_2 = cv2.imread(args.im2)

    # list_pairs_matched_keypoints = extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_test=0.7)

    # ===== create a panorama image
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=args.output, img=img_panorama.clip(0.0, 255.0).astype(np.uint8))
