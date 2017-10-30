'''
  File name: morph_tri.py
  Author: Devesh Dayal
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import math

def computeAffine(a, b):
    i = np.ones((3, 3))
    j = np.ones((3, 3))
    i[:, :-1] = a
    j[:, :-1] = b
    return np.dot(j.T, np.linalg.inv(i.T))

def morph(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    morphed_frame = np.empty(im2.shape, dtype=np.uint8)
    H, W, three = im2.shape

    tri = Delaunay((1-warp_frac) * im1_pts + warp_frac * im2_pts)

    pixel_pos = np.asarray([[x, y] for x in range(H) for y in range(W)])
    triangle_indices = tri.find_simplex(pixel_pos)

    triangle_point_map = {x: np.matrix([[1], [1], [1]]) for x in range(len(triangle_indices))}

    for i in range(len(triangle_indices)):
      triangle_idx = triangle_indices[i]
      if triangle_idx != -1:
        vector = np.matrix([[pixel_pos[i][0]], [pixel_pos[i][1]], [1]])
        triangle_point_map[triangle_idx] = np.concatenate((triangle_point_map[triangle_idx], vector), axis=1)
    
    for i in range(len(tri.simplices)):
        triangle = tri.simplices[i]
        src = im1_pts[triangle]
        tgt = im2_pts[triangle]
        intermediate = (1-warp_frac) * src + warp_frac * tgt

        A = computeAffine(intermediate, src)
        B = computeAffine(intermediate, tgt)

        # compute the points that lie inside this current triangle
        points_in_triangle = triangle_point_map[i][:, 1:]

        # multiply with the affines
        A = np.dot(A, points_in_triangle).astype(int)
        B = np.dot(B, points_in_triangle).astype(int)

        A[0] = np.clip(A[0], 0, H-1)
        A[1] = np.clip(A[1], 0, W-1)
        B[0] = np.clip(B[0], 0, H-1)
        B[1] = np.clip(B[1], 0, W-1)

        src_rgb = im1[A[0], A[1]]
        tgt_rgb = im2[B[0], B[1]]

        morphed_frame[points_in_triangle[0, :], points_in_triangle[1, :]] = (1 - dissolve_frac) * src_rgb + dissolve_frac * tgt_rgb
    
    return morphed_frame

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  im1_pts = np.asarray(im1_pts)
  im2_pts = np.asarray(im2_pts)
  H, W, three = im2.shape
  # shape of final result
  morphed_im = np.zeros((len(warp_frac), H, W, 3), dtype=np.uint8)
  # per frame computation
  for frame in range(len(warp_frac)):
    # print "processing frame: " + str(frame)
    morphed_im[frame] = morph(im1, im2, im1_pts, im2_pts, warp_frac[frame], dissolve_frac[frame])
  return morphed_im



# OLD CODE BELOW



# def morph_tri2(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
#   # make sure inputs are numpy arrays
#   im1_pts = np.asarray(im1_pts)
#   im2_pts = np.asarray(im2_pts)

#   # this is the intermediate mean matrix of points
#   tri = Delaunay(0.5 * im1_pts + 0.5 * im2_pts)

#   # Code Parameters
#   NUM_TRIANGLES = len(tri.simplices)
#   m = len(warp_frac)
#   H, W, three = im1.shape
#   morphed_im = np.zeros((m, H, W, 3), dtype=np.uint8)
#   src_colors = np.copy(im1)
#   tgt_colors = np.copy(im2)

#   # calculate the triangle indices for each pixel position
  # pixel_pos = np.asarray([[x, y] for x in range(H) for y in range(W)])
  # triangle_indices = tri.find_simplex(pixel_pos)

#   # calculating the matrices for im1_pts and im2_pts
#   src_matrix = np.zeros((NUM_TRIANGLES, 3, 3))
#   tgt_matrix = np.zeros((NUM_TRIANGLES, 3, 3))
#   for idx in range(NUM_TRIANGLES):
#     # source matrix
#     t = im1_pts[tri.simplices][idx]
#     mat = np.matrix([[t[0][0], t[1][0], t[2][0]], [t[0][1], t[1][1], t[2][1]], [1, 1, 1]])
#     src_matrix[idx] = mat

#     # target matrix
#     t = im2_pts[tri.simplices][idx]
#     mat = np.matrix([[t[0][0], t[1][0], t[2][0]], [t[0][1], t[1][1], t[2][1]], [1, 1, 1]])
#     tgt_matrix[idx] = mat
    

#   # iterate over each frame
#   for frame in range(m):
#     print "Processing frame: " + str(frame)
#     wf = warp_frac[frame]
#     df = dissolve_frac[frame]

#     intermediate_shape = (1.0 - wf) * im1_pts + wf * im2_pts

#     # calculate the inverse matrices for the target points
#     intermediate_matrix_inv = np.zeros((NUM_TRIANGLES, 3, 3))
#     for idx in range(NUM_TRIANGLES):
#       t = intermediate_shape[tri.simplices][idx]
#       m = np.matrix([[t[0][0], t[1][0], t[2][0]], [t[0][1], t[1][1], t[2][1]], [1, 1, 1]])
#       intermediate_matrix_inv[idx] = np.linalg.inv(m)

#     # calculate barycentric coordinates and new grid values
#     # src_points = np.zeros((H * W, 2, 1), dtype=np.int32)
#     # tgt_points = np.zeros((H * W, 2, 1), dtype=np.int32)
    
#     src_points = np.zeros((H * W, 2, 1))
#     tgt_points = np.zeros((H * W, 2, 1))

#     # use original grid values for points not in triangles
#     original_group_idx = np.where(triangle_indices == -1)
#     original_group = pixel_pos[original_group_idx]
#     for i in range(len(original_group)):
#       point = original_group[i]
#       src_points[original_group_idx[0][i]] = np.asarray(point).reshape((2, 1))
#       tgt_points[original_group_idx[0][i]] = np.asarray(point).reshape((2, 1))

#     for i in range(NUM_TRIANGLES):
#       group_idx = np.where(triangle_indices == i)
#       group_values = pixel_pos[group_idx]
#       num_points = len(group_values)
#       triangle_group = np.matrix([group_values[:, 0], group_values[:, 1], np.ones(num_points)])

#       barycentric = np.dot(intermediate_matrix_inv[i], triangle_group)
#       src_point_group = np.dot(src_matrix[i], barycentric)
#       tgt_point_group = np.dot(tgt_matrix[i], barycentric)

#       src_point_group[:2, :] = (src_point_group[:2, :]*1.0) / src_point_group[2, :]
#       tgt_point_group[:2, :] = (tgt_point_group[:2, :]*1.0) / tgt_point_group[2, :]

#       for i in range(num_points):
#         src_val = src_point_group[:2, i]
#         src_bounded_val = np.asarray([[ceil(min(src_val[0], H - 1)], [min(src_val[1], W - 1)]]).reshape((2, 1))
#         src_points[group_idx[0][i]] = src_bounded_val

#         tgt_val = tgt_point_group[:2, i]
#         tgt_bounded_val = np.asarray([[min(tgt_val[0], H - 1)], [min(tgt_val[1], W - 1)]]).reshape((2, 1))
#         tgt_points[group_idx[0][i]] = tgt_bounded_val
    

#     # for idx in range(H * W):
#     #   point = pixel_pos[idx]
#     #   target_triangle_index = triangle_indices[idx]
#     #   # ignore values that don't fit into the triangle
#     #   if target_triangle_index != -1.0:
#     #     point_vec = np.array([[point[0]], [point[1]], [1]])
#     #     barycentric = np.dot(intermediate_matrix_inv[target_triangle_index], point_vec)

#     #     # calculate warp from source to intermediate
#     #     src_point = np.dot(src_matrix[target_triangle_index], barycentric)
#     #     if src_point[2] != 0:
#     #       src_point[0] = min(max((src_point[0]*1.0) / src_point[2], 0), H - 1)
#     #       src_point[1] = min(max((src_point[1]*1.0) / src_point[2], 0), W - 1)
#     #     src_points[idx] = src_point[:2]

#     #     # calculate warp from target to intermediate
#     #     tgt_point = np.dot(tgt_matrix[target_triangle_index], barycentric)
#     #     if tgt_point[2] != 0:
#     #       tgt_point[0] = min(max((tgt_point[0]*1.0) / tgt_point[2], 0), H - 1)
#     #       tgt_point[1] = min(max((tgt_point[1]*1.0) / tgt_point[2], 0), W - 1)
#     #     tgt_points[idx] = tgt_point[:2]
#     #   else:
#     #     src_points[idx] = np.asarray(point).reshape((2, 1))
#     #     tgt_points[idx] = np.asarray(point).reshape((2, 1))

#     # bound the results
#     src_points[np.where(src_points < 0)] = 0
#     tgt_points[np.where(tgt_points < 0)] = 0

#     # get RGB values from the two warp point positions
#     src_rgb = im1[src_points[:, 0], src_points[:, 1]]
#     tgt_rgb = im2[tgt_points[:, 0], tgt_points[:, 1]]

#     # cross dissolve onto a copy of the base image
#     for idx in range(H * W):
#       # set each RGB value
#       src_colors[src_points[idx, 0], src_points[idx, 1]] = src_rgb[idx]
#       tgt_colors[tgt_points[idx, 0], tgt_points[idx, 1]] = tgt_rgb[idx]

#     frame_result = (1.0 - df) * src_colors + df * tgt_colors
#     morphed_im[frame] = frame_result

#   return morphed_im