import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.signal

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy import ndimage

import sol4_utils

deriv_vec = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])

# as pdf states, we blur with ker size of 3
KERNEL_SIZE = 3
# we want 7x7 matricies, therefore desc_rad is set to 3
DESC_RAD = 3
#TODO: change or make sure its suppose to be consts
M_SIZE = 7
N_SIZE = 7
RADIUS_SIZE = 4
BAD_IDX = -1

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # setting up k var as instructed in exercise
    k = 0.04
    # calculating deriv in x and y orientation
    dx = ndimage.filters.convolve(im, deriv_vec, mode='nearest')
    dy = ndimage.filters.convolve(im, np.transpose(deriv_vec), mode='nearest')
    #TODO: check if multiplication is element-wise (np.multiply) or regular (np.dot)
    dx_pow = sol4_utils.blur_spatial(np.multiply(dx, dx), KERNEL_SIZE)
    dy_pow = sol4_utils.blur_spatial(np.multiply(dy, dy), KERNEL_SIZE)
    dx_dy_mul = sol4_utils.blur_spatial(np.multiply(dx, dy), KERNEL_SIZE)
    dy_dx_mul = sol4_utils.blur_spatial(np.multiply(dy, dx), KERNEL_SIZE)
    # calculating trace and it's power
    trace = np.add(dx_pow, dy_pow)
    trace_pow = np.multiply(trace, trace)
    # calculating determinant of matrix M
    det_ad = np.multiply(dx_pow, dy_pow)
    det_bc = np.multiply(dx_dy_mul, dy_dx_mul)
    det_M = np.subtract(det_ad, det_bc)
    # calculating response for each pixel in the image
    response_im = np.subtract(det_M, k * trace_pow)

    max_response = non_maximum_suppression(response_im)

    return np.stack(np.flip(np.where(max_response), axis=0), axis=-1)

def helper_sample(x, y, desc_rad):
    div_factor = 1/4
    vec = np.arange(-desc_rad, desc_rad + 1)
    vec_x = np.tile(np.add(vec, x * div_factor), 1 + desc_rad * 2)
    vec_y = np.repeat(np.add(vec, y * div_factor), 1 + desc_rad * 2)
    # print(vec_x)
    # print(vec_y)
    return vec_x, vec_y

def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = 1 + desc_rad * 2
    N = pos.shape[0]
    output_array = np.zeros((N, K, K), dtype=np.float64)
    for i in range(N):
        # VERY IMPORTANT: helper sample x,y coords are flipped, so 0'th coord is y, 1'st coord is x:
        temp_x, temp_y = helper_sample(pos[i,1], pos[i,0], desc_rad)
        mat = scipy.ndimage.interpolation.map_coordinates(im, [temp_x, temp_y], order=1, prefilter=False)
        mat = np.reshape(mat, (K, K))
        # checking that we're not dividing by 0
        if (np.linalg.norm(mat - np.mean(mat)) != 0):
            mat = np.divide((mat - np.mean(mat)), (np.linalg.norm(mat - np.mean(mat))))
        output_array[i] = mat
    # print(output_array.shape[0], "x", output_array.shape[1], "x", output_array.shape[2])
    return output_array

def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
    """
    feature_pts = spread_out_corners(pyr[0], M_SIZE, N_SIZE, RADIUS_SIZE)
    feature_descriptor = sample_descriptor(pyr[2], feature_pts, DESC_RAD)
    return feature_pts, feature_descriptor


def desc_score_helper(desc1, feature_desc, min_score):
    """
    gets a feature descriptor array with shape (N,K,K) and calculate it's score with the input feature descriptor
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param n: A feature descriptor to calculate it's dot product with
    :return: the indices of the 2 best scores
    """
    score_list = np.zeros(shape=(desc1.shape[0]))
    feature_desc_flatten = feature_desc.flatten()
    for i in range(desc1.shape[0]):
        score_list[i] = np.dot(desc1[i].flatten(), feature_desc_flatten)
    best_indices = np.argpartition(score_list, -2)[-2:]
    for i in range(best_indices.shape[0]):
        if score_list[best_indices[i]] <= min_score:
            best_indices[i] = BAD_IDX
    return best_indices

def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    best_scores_desc1 = np.zeros(shape=(desc1.shape[0], 2), dtype=np.int64)
    best_scores_desc2 = np.zeros(shape=(desc2.shape[0], 2), dtype=np.int64)
    matching_indices_desc1 = np.array([])
    matching_indices_desc2 = np.array([])

    # getting best matches for each desc2 feature
    for i in range(desc2.shape[0]):
        best_scores_desc2[i] = desc_score_helper(desc1, desc2[i], min_score)

    for i in range(desc1.shape[0]):
        best_scores_desc1[i] = desc_score_helper(desc2, desc1[i], min_score)

    # comparing each feature's bests to find matches
    for i in range(best_scores_desc2.shape[0]):
        for j in range(best_scores_desc2.shape[1]):
            # iterating over 2 best scores of desc2[i] descriptor
            idx = best_scores_desc2[i][j]
            if idx == BAD_IDX:
                continue
            if i in best_scores_desc1[idx]:
                matching_indices_desc1 = np.append(matching_indices_desc1, idx)
                matching_indices_desc2 = np.append(matching_indices_desc2, i)
    return matching_indices_desc1, matching_indices_desc2


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    mat = np.zeros(shape=(pos1.shape[0], pos1.shape[1]))
    for i in range(pos1.shape[0]):
        x = pos1[i][0]
        y = pos1[i][1]
        temp_vec = np.dot(H12, np.array([x, y, 1]))
        if temp_vec[2] == 0:
            # z tilda is 0, cant divide with him
            print("error z2=0")
            continue
        mat[i] = [temp_vec[0] / temp_vec[2], temp_vec[1] / temp_vec[2]]
    return mat


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    N = points1.shape[0]
    cur_max_inliners = np.array([], dtype=np.int64)

    for i in range(num_iter):
        temp_best_inliners = np.array([], dtype=np.int64)
        # rolling 2 random numbers in N range (as index), and getting the points
        random_pts = np.random.choice(N, 2)
        points1_temp = elements_from_indices(points1, random_pts)
        points2_temp = elements_from_indices(points2, random_pts)
        # estimating the homography using those points
        H12 = estimate_rigid_transform(points1_temp, points2_temp, translation_only)
        # calculating new homography based on
        points1_to_2 = apply_homography(points1, H12)
        # calculating for each descriptor in new_homo the euclidean error

        for j in range(points1_to_2.shape[0]):
            if np.linalg.norm(points1_to_2[j] - points2[j]) ** 2 < inlier_tol:
                # inliner match
                temp_best_inliners = np.append(temp_best_inliners, j)
        # checking if current iteration yielded max number of inliners
        if temp_best_inliners.shape[0] > cur_max_inliners.shape[0]:
            cur_max_inliners = temp_best_inliners
    print("max inliner: ", cur_max_inliners.shape[0])

    print("cur max inliners: ", cur_max_inliners)

    final_inliners1 = elements_from_indices(points1, cur_max_inliners)
    final_inliners2 = elements_from_indices(points2, cur_max_inliners)

    print(final_inliners1)
    print()
    print(final_inliners2)

    H12 = estimate_rigid_transform(final_inliners1, final_inliners2, translation_only)
    points1_to_2 = apply_homography(points1, H12)
    final = np.array([], dtype=np.int64)
    for j in range(points1_to_2.shape[0]):
        if np.linalg.norm(points1_to_2[j] - points2[j]) ** 2 < inlier_tol:
            final = np.append(final, j)
    print("final inliners found: ", final.shape[0])
    return H12, final


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    conc_img = np.concatenate((im1, im2))

    points2_x, points2_y = np.dsplit(points2, points2.shape[1])
    points2_x += im1.shape[1]
    points2_y += im1.shape[0]
    points2 = np.concatenate((points2_x, points2_y))
    plt.imshow(conc_img, cmap='gray')
    for i in range(points2.shape[0]):
        plt.plot(points1[i, 0],points1[i, 1], points2[i, 0], points2[i, 1], marker='o')
    inliers1 = np.take(points1, inliers)
    inliers2 = np.take(points2, inliers)


    plt.show()


def accumulate_homographies(H_succesive, m):
  """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography 
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """ 
  pass


def compute_bounding_box(homography, w, h):
  """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
  pass


def warp_channel(image, homography):
  """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
  pass


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) & 
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]


  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()

def elements_from_indices(arr, indices):

    return arr[indices.astype(np.int32)]

    matching_coords = np.zeros(shape=(indices.shape[0], 2))
    for i in range(indices.shape[0]):
        idx = int(indices[i])
        matching_coords[i] = arr[idx]
    return matching_coords


im1 = sol4_utils.read_image("C:\/Users\Imri\PycharmProjects\IP_ex4\ex4-imrilu\external\oxford1.jpg", 1)
im2 = sol4_utils.read_image("C:\/Users\Imri\PycharmProjects\IP_ex4\ex4-imrilu\external\oxford2.jpg", 1)
pyr1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)[0]
pyr2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)[0]
desc_coords1, desc1 = find_features(pyr1)
desc_coords2, desc2 = find_features(pyr2)
matching_idx1, matching_idx2 = match_features(desc1, desc2, 0.5)


print("hello")

ransac_homography(elements_from_indices(desc_coords1, matching_idx1), elements_from_indices(desc_coords2, matching_idx2), 1000, 10)

# R = spread_out_corners(im1, 7,7,3)
#
# plt.imshow(im1, cmap='gray')
# plt.plot(R[:,0], R[:,1], "o")
# plt.show()

