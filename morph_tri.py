'''
  File name: morph_tri.py
  Author: Krishna Bharathala
  Date created: 10/09/2017
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

import numpy as np
from scipy.spatial import Delaunay

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):

  ## get the height and width of the image
  ## TODO(make sure they are the same shape, else do some padding)
  H, W = im1.shape[0], im1.shape[1]

  morphed_im = np.zeros((len(warp_frac), H, W, 3), dtype=np.uint8)
  warped = np.zeros(im1_pts.shape)

  ## for each frame calculate the change.
  for frame_num in range(len(warp_frac)):
    morphed_frame = np.copy(im1).astype(np.uint8)

    ## calculate the intermediate triangle shape
    int_pts = im1_pts * (1 - warp_frac[frame_num]) + im2_pts * warp_frac[frame_num]
    warped = int_pts
    int_triangles_s = Delaunay(int_pts)
    int_triangles = int_pts[int_triangles_s.simplices]

    ## for the src and the tgt calculate at what values each triangle lies
    src_triangles = im1_pts[int_triangles_s.simplices]
    tgt_triangles = im2_pts[int_triangles_s.simplices]

    ## calculate the inverse matrix for each of the intermediate triangles.
    int_triangle_inverses = [np.linalg.inv([[t[0][0], t[1][0], t[2][0]], [t[0][1], t[1][1], t[2][1]], [1, 1, 1]]) for t in int_triangles]

    ## create a dictionary to store all the points that are in the triangle
    triangle_points_dictionary = {x:np.matrix([[1], [1], [1]]) for x in range(-1, len(int_triangles))}

    ## get a list of all the pixels
    pixel_positions = [[x, y] for x in range(H) for y in range(W)]

    ## figure out what triangle each pixel is in
    pixel_triangle_index = int_triangles_s.find_simplex(pixel_positions)

    ## add the pixel to the appropriate array in triangle_points_matrix
    for idx, index in enumerate(pixel_triangle_index):
      matrix = np.matrix([[pixel_positions[idx][0]], [pixel_positions[idx][1]], [1]])
      triangle_points_dictionary[index] = np.concatenate((triangle_points_dictionary[index], matrix), axis=1)

    ## iterate over each triangle
    for triangle_num in range(len(int_triangle_inverses)):
      triangle_points = triangle_points_dictionary[triangle_num][:,1:]
      bary_center = np.dot(int_triangle_inverses[triangle_num], triangle_points)

      ## Calculating the src transform
      src = src_triangles[triangle_num]
      src_m = np.matrix([[src[0][0], src[1][0], src[2][0]], [src[0][1], src[1][1], src[2][1]], [1, 1, 1]])
      src_pt = np.dot(src_m, bary_center)
      src_pt = (src_pt / src_pt[2])[:2].astype(int)

      src_pt[0, :] = np.clip(src_pt[0, :], 0, H-1)
      src_pt[1, :] = np.clip(src_pt[1, :], 0, W-1)

      ## Calculating the tgt transform
      tgt = tgt_triangles[triangle_num]
      tgt_m = np.matrix([[tgt[0][0], tgt[1][0], tgt[2][0]], [tgt[0][1], tgt[1][1], tgt[2][1]], [1, 1, 1]])
      tgt_pt = np.dot(tgt_m, bary_center)
      tgt_pt = (tgt_pt / tgt_pt[2])[:2].astype(int)

      tgt_pt[0, :] = np.clip(tgt_pt[0, :], 0, H-1)
      tgt_pt[1, :] = np.clip(tgt_pt[1, :], 0, W-1)

      src_color = im1[src_pt[0], src_pt[1]]
      tgt_color = im2[tgt_pt[0], tgt_pt[1]]

      ## Calculate the dissolved color
      final_color = ((1-dissolve_frac[frame_num]) * src_color + dissolve_frac[frame_num] * tgt_color)

      ## set the morphed_frame pixel value to the new color
      morphed_frame[triangle_points[0, :], triangle_points[1, :]] = final_color

    morphed_im[frame_num] = morphed_frame.astype(np.uint8)

  return morphed_im, warped
  