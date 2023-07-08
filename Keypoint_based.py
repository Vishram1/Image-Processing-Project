# This Whole code is written and executed by Vishwaram Reddy EE20BTECH11059.

import os
import numpy as np
import cv2 as cv
# from SIFT_implementation import computeKeypointsAndDescriptors  if using this function make sure to free up all the
# memory.
from sklearn.cluster import AgglomerativeClustering

directory = "E:\Image Project"   # Change the directory as needed.
Test_images = [f for f in os.listdir(directory) if f.endswith('.jpg')]
true_pos = 0
false_pos = 0
original_images = 8
tampered_images = 32

for image in Test_images:

    ori = cv.imread(os.path.join(directory, image), cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create(nfeatures=9000)   # This nfeatures part should be changed accordingly if using this method.
    k, d = sift.detectAndCompute(ori, None)

# Using the keypoint distance algorithm mentioned in the paper.
    def KeyPoint_Distance(keypoints=k, descriptors=d):
        descriptors = np.array(descriptors)
        keypoints = np.array(keypoints)
        key_dist = []
        diff = descriptors[:, np.newaxis] - descriptors   # Could Eat up a lot of memory be careful.
        for i in range(len(diff)):
            key_dist.append(np.sqrt(np.sum(np.square(diff[i]), axis=1)))
        key_dist = np.array(key_dist)
        kp_dist = []
        for i in range(len(key_dist)):
            kp_dist.append(np.delete(key_dist[i], i))
        kp_dist = np.array(kp_dist)
        matched_kp = {}
        for i in range(len(keypoints)):
            a = np.sort(kp_dist[i])
            tmp = []
            for j in range(len(a) - 1):
                r = a[j] / a[j + 1]
                if r < 0.6:
                    indx = np.where(kp_dist[i] == a[j])
                    tmp.append(keypoints[indx[0][0]])
            matched_kp[k[i]] = tmp

        return matched_kp


    mc = KeyPoint_Distance()
    seto = set()

    for i in mc:
        if len(mc[i]) > 0:
            x, y = i.pt
            for j in mc[i]:
                p, q = j.pt
                seto.add((p, q))
            seto.add((x, y))

    matched_points = np.array(list(seto))

# Using Aggloromative clustering to combine the similar parts of the image and detect copy move based on the criteria
    # given in the paper.
    def detect_copy_move_attack(keypoints=matched_points, Th=1.6):
        clustering = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='ward',
                                             distance_threshold=Th).fit(keypoints)

        labels = clustering.labels_
        cluster_sizes = np.bincount(labels)
        valid_clusters = np.where(cluster_sizes >= 3)[0]

        distances = np.zeros((len(valid_clusters), len(valid_clusters)))
        for i in range(len(valid_clusters)):
            for j in range(i + 1, len(valid_clusters)):
                cluster1_mask = (labels == valid_clusters[i])
                cluster2_mask = (labels == valid_clusters[j])
                cluster1_center = np.mean(keypoints[cluster1_mask], axis=0)
                cluster2_center = np.mean(keypoints[cluster2_mask], axis=0)
                dist = np.sqrt(np.sum((cluster1_center - cluster2_center) ** 2))
                reciprocal_dist = 1.0 / dist if dist > 1e-6 else np.inf
                distances[i, j] = reciprocal_dist
                distances[j, i] = reciprocal_dist

        for i in range(len(valid_clusters)):
            for j in range(i + 1, len(valid_clusters)):
                if distances[i, j] >= 1.0 / Th:
                    cluster1_mask = (labels == valid_clusters[i])
                    cluster2_mask = (labels == valid_clusters[j])
                    num_pairs = 0
                    for x in range(len(keypoints)):
                        if cluster1_mask[x]:
                            for y in range(len(keypoints)):
                                if cluster2_mask[y] and (np.abs(keypoints[x, 0] - keypoints[y, 0]) < 1e-6) and (
                                        np.abs(keypoints[x, 1] - keypoints[y, 1]) < 1e-6):
                                    num_pairs += 1
                    if num_pairs >= 3:
                        return True

        return False


    if "tamp" in image and detect_copy_move_attack():
        true_pos += 1
    elif "tamp" not in image and not detect_copy_move_attack():
        true_pos += 1
    elif "tamp" not in image and detect_copy_move_attack():
        false_pos += 1

True_Positive_Rate = true_pos / tampered_images
False_Postive_Rate = false_pos / original_images

print("TPR:- ", True_Positive_Rate * 100)
print("FPR:- ", False_Postive_Rate * 100)
