import argparse
import cv2
import mediapipe as mp
import os
from datetime import datetime
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

# Extended ODST Pipeline

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# TODO: This could be made into a cmd ln arg?
EXEMPLAR_DIRS = {
    "vertumnus": "src/arcimboldo_exemplars/vertumnus",
    "four_seasons": "src/arcimboldo_exemplars/four_seasons",
    "floral": "src/arcimboldo_exemplars/floral",
    "summer": "src/arcimboldo_exemplars/summer"
}

def euclidian_dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_masks_and_com(segmentation):
    colours = np.unique(segmentation.reshape(-1,3), axis=0)
    n_col = len(colours)

    masks = []
    com = []
    for i, c in enumerate(colours):
        if np.all(c == np.array([0, 0, 0])):
            continue

        # Extract mask
        m = np.all(segmentation == c, axis=-1).astype(np.uint8)
        masks.append(m)

        # Get mask Center of Mass (COM)
        mass_y, mass_x = np.where(m == True)
        cx = np.average(mass_x); cy = np.average(mass_y)
        cx = int(cx); cy = int(cy)

        com.append((cx, cy))

    return masks, com

def get_nearest_landmarks(landmarks, pts, image_bounds):
    b_x, b_y = image_bounds

    nearest_idxs = []
    for pt in pts:

        best_pt_idx = None
        best_dist = float('inf')
        # TODO: Landmarks could be spatially partitioned (e.g. KD-tree) to make this faster.
        for i, l in enumerate(landmarks.multi_face_landmarks[0]):
            idx = i + 6 # Compensate for 0-5 reserved for face keypoints.

            screen_space_coord = _normalized_to_pixel_coordinates(l.x, l.y, b_x, b_y)
            dist = euclidian_dist(pt, screen_space_coord)

            if dist < best_dist:
                best_dist = dist
                best_pt_idx = idx

        nearest_idxs.append(best_pt_idx)
    
    return nearest_idxs

def get_mesh_x_rotation(mesh, bounds):
    b_x, b_y = bounds

    pts_3d = []
    pts_2d = []
    for face_landmarks in mesh.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            # Selection of keypoints from the front of the face
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                x, y = int(lm.x * b_x), int(lm.y * b_y)
                pts_2d.append([x, y])
                pts_3d.append([x, y, lm.z])
        
        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)

    focal_length = 1 * b_x
    K = np.array([ [focal_length, 0, b_y / 2],
                            [0, focal_length, b_x / 2],
                            [0, 0, 1]])

    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    _, rot_vec, _ = cv2.solvePnP(pts_3d, pts_2d, K, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    y = angles[1] * 360

    return y

def landmark_to_screenspace(landmark, bounds):
    b_x, b_y = bounds
    coord = _normalized_to_pixel_coordinates(landmark.x, landmark.y, b_x, b_y)

    return coord

def add_extra_landmarks(mesh, img_bounds):
    base_landmarks = list(mesh.multi_face_landmarks[0].landmark)
    extra_landmarks = get_image_bounding_landmarks(3, img_bounds)

    combined = base_landmarks + extra_landmarks

    return combined
def create_norm_landmark(x, y, z, bounds):
    b_x, b_y = bounds
    l = NormalizedLandmark()
    l.x = x / b_x; l.y = y / b_y; l.z = z

    return l

def get_image_bounding_landmarks(n, bounds):
    # TODO: N != 3 is currently not supported in the triangulation.
    if n < 3:
        n = 3

    Z_COORD = 10 # Set very low to ensure drawn first
    b_x, b_y = bounds
    # Corners
    landmarks = [
        create_norm_landmark(0, 0, Z_COORD, bounds),
        create_norm_landmark(0, b_y, Z_COORD, bounds),
        create_norm_landmark(b_x, 0, Z_COORD, bounds),
        create_norm_landmark(b_x, b_y, Z_COORD, bounds)
    ]

    # In-betweens
    horizontal = [create_norm_landmark(int(d * b_x / (n - 1)), v, Z_COORD, bounds) 
                  for d in range(1, n-1) for v in [0, b_y]]
    vertical   = [create_norm_landmark(h, int(d * b_y / (n - 1)), Z_COORD, bounds) 
                  for d in range(1, n-1) for h in [0, b_x]]

    landmarks.extend(horizontal + vertical)

    return landmarks

def get_face_mesh_triangulation(bounds, n_border_landmarks=3):

    edgemap = set(mp_face_mesh.FACEMESH_TESSELATION)

    # See below for keypoint visualization and definition
    # https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    # https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    extra_edges = set([
        # Left eye covers
        (173, 155), (157, 155), (154, 157), (158, 154), (153, 158),
        (159, 153), (145, 159), (160, 145), (144, 160), (161, 144),
        (163, 161), (246, 163), (  7, 246),

        # Right eye covers
        (249, 466), (390, 466), (388, 390), (373, 388), (373, 387),
        (374, 387), (386, 374), (380, 386), (385, 380), (381, 385),
        (384, 381), (382, 384), (398, 382),

        # Mouth Cover
        (308, 415), (308, 324), (415, 324), (324, 310), (310, 318),
        (318, 311), (311, 402), (402, 312), (312, 317), (317,  13),
        ( 13,  14), ( 13,  87), ( 87,  82), ( 82, 178), ( 81, 178),
        ( 88,  81), ( 80,  88), ( 95,  80), (191,  95),

        # Iris tracking can be used over eye covers (although the mesh might not have nice triangulation).
        # right eye pupil -> iris
        # (473, 474), (473, 475), (473, 476), (473, 477),
        # left eye pupil -> iris
        # (468, 469), (468, 470), (468, 471), (468, 472),

        # Assuming n extra is 3. (There is probably a better way of doing this if
        #   we accumulate all points on the contour of the face.)
        # Note: Extra landmarks indexing starts at 478
        # top left (478)
        (478,  21), (478,  54), (478, 103), (478,  67), (478, 162),
        # bottom left (479)
        (479,  58), (479, 172), (479, 136), (479, 150), (479, 149),
        # top right (480)
        (480, 297), (480, 332), (480, 284), (480, 251), (480, 389),
        # bottom right (481)
        (481, 288), (481, 397), (481, 365), (481, 379), (481, 378),
        # top middle (482)
        (482, 109), (482,  10), (482, 338), (482,  67), (482, 297),
        # bottom middle (483)
        (483, 176), (483, 148), (483, 152), (483, 377), (483, 400), (483, 149), (483, 378),
        # left middle (484)
        (484, 162), (484, 127), (484, 234), (484,  93), (484, 132), (484,  58),
        # right middle (485)
        (485, 389), (485, 356), (485, 454), (485, 323), (485, 361), (485, 288),

        # Join the extra landmarks at the border:
        (478, 482), (478, 484), # TL 
        (480, 482), (480, 485), # TR
        (479, 484), (479, 483), # BL
        (481, 483), (481, 485), # BR
    ])

    edgemap = edgemap.union(extra_edges)

    vertex_edge_list = {}
    for edge in edgemap:
        a, b = edge

        # a -> b
        if a in vertex_edge_list:
            if not b in vertex_edge_list[a]:
                vertex_edge_list[a].append(b)
        else:
            vertex_edge_list[a] = [b]

        # b -> a
        if b in vertex_edge_list:
            if not a in vertex_edge_list[b]:
                vertex_edge_list[b].append(a)
        else:
            vertex_edge_list[b] = [a]

    faces = {}
    for vertex in vertex_edge_list:
        neighbours = vertex_edge_list[vertex]

        for i, n in enumerate(neighbours):
            for j, m in enumerate(neighbours):

                if i == j:
                    continue

                if m in vertex_edge_list[n]:
                    faces[(vertex, n, m)] = True

    return faces

def get_triangle(triangle_lst, mesh, triangle, bounds):
    tri = np.array([
        landmark_to_screenspace(triangle_lst[triangle[0]], bounds),
        landmark_to_screenspace(triangle_lst[triangle[1]], bounds),
        landmark_to_screenspace(triangle_lst[triangle[2]], bounds)
    ])

    return tri

def furthest_triangle_z(triangle_lst, triangle):
    # The mesh triangles for a face are smooth, so this painters algorithm approach should work fine.
    # Note min() as z goes negative into the screen space.
    return min(
        triangle_lst[triangle[0]].z,
        triangle_lst[triangle[1]].z,
        triangle_lst[triangle[2]].z
    )


# The following two util functions are sourced from:
# https://github.com/spmallick/PyImageConf2018/blob/master/faceBlendCommon.py

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size):

  # Given a pair of triangles, find the affine transform.
  warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

  # Apply the Affine Transform just found to the src image
  dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

  return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = apply_affine_transform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def transform(img1, mesh1, mesh2, out):
    bounds = out.shape[:2][::-1]
    triangles = get_face_mesh_triangulation(bounds)

    bounds1 = img1.shape[0:2][::-1] # Reverse to w, h
    bounds2 = out.shape[0:2][::-1] # Reverse to w, h

    tri_lst1 = add_extra_landmarks(mesh1, bounds1)
    tri_lst2 = add_extra_landmarks(mesh2, bounds2)

    # Painter's algorithm
    sorted_triangles = sorted(triangles.keys(), reverse=True, key=lambda t: furthest_triangle_z(tri_lst2, t))
    for t in sorted_triangles:
        # TODO: We should do this from triangles furthest back in mesh2 first?
        tri1 = get_triangle(tri_lst1, mesh1, t, bounds1)
        tri2 = get_triangle(tri_lst2, mesh2, t, bounds2)

        # Piecewise Affine Transform
        warp_triangle(img1, out, tri1, tri2)

    return np.uint8(out)

# https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
def visualize_landmarks(img, results):
    if not results.multi_face_landmarks:
        return img
    
    mp_drawing = mp.solutions.drawing_utils
    annotated_image = img.copy()
    for face_landmarks in results.multi_face_landmarks:
    #   print(face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
      
    return annotated_image

def vis_landmark_pts(img, mesh, show=False):
    out = img.copy()
    bounds = out.shape[:2][::-1]

    tri_lst = add_extra_landmarks(mesh, bounds)

    for i, v in enumerate(tri_lst):
        coord = landmark_to_screenspace(v, bounds)
        cv2.circle(out, coord, 4, (0, 255, 0), thickness=-1)
        # cv2.putText(out, str(i), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))

    if show:
        cv2.imshow("landmark labels", out)
        # cv2.imwrite("./landmark-vis.png", out)
        cv2.waitKey(0)

    return out

def vis_mesh_faces(img, mesh, show=False):

    ref = img.copy()
    prng = np.random.RandomState(50)

    bounds = ref.shape[:2][::-1] # Reverse to w, h
    triangles = get_face_mesh_triangulation(bounds)
    tri_lst = add_extra_landmarks(mesh, bounds)

    sorted_triangles = sorted(triangles.keys(), reverse=True, key=lambda t: furthest_triangle_z(tri_lst, t))
    for t in sorted_triangles:
        color = prng.random(3) * 255
        tri = get_triangle(tri_lst, mesh, t, bounds)
        cv2.drawContours(ref, [tri], 0, color, -1)

    if show:
        cv2.imshow("triangles", ref)
        cv2.waitKey(0)

    return ref

def get_face_mesh(img, visualise=False):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if visualise:
            vis = visualize_landmarks(img, results)
            cv2.imshow('vis', vis)
            cv2.waitKey(0)

    return results

def load_ref_and_style(transfer_type, flip=False):

    exemplar_dir = EXEMPLAR_DIRS[transfer_type]

    ref         = cv2.imread(os.path.join(exemplar_dir, "ref.png"))
    seg         = cv2.imread(os.path.join(exemplar_dir, "ref_sem.png"))
    style       = cv2.imread(os.path.join(exemplar_dir, "style.png"))
    style_sem   = cv2.imread(os.path.join(exemplar_dir, "style_sem.png"))

    if flip:
        ref = cv2.flip(ref, 1)
        seg = cv2.flip(seg, 1)
        style = cv2.flip(style, 1)
        style_sem = cv2.flip(style_sem, 1)

    return ref, seg, style, style_sem

def main(transfer_type, content_path, no_flip=False, show=False):

    content = cv2.imread(content_path)
    content_bounds = content.shape[:2][::-1]
    content_mesh = get_face_mesh(content, visualise=False)
    content_rot = get_mesh_x_rotation(content_mesh, content_bounds)

    ref, seg, style, style_sem = load_ref_and_style(transfer_type, flip=False)
    ref_bounds = ref.shape[:2][::-1]
    ref_mesh = get_face_mesh(ref, visualise=False)
    ref_rot = get_mesh_x_rotation(ref_mesh, ref_bounds)

    if (not no_flip) and content_rot * ref_rot < 0:
        # They have different rotations: reload refs and flip them.
        ref, seg, style, style_sem = load_ref_and_style(transfer_type, flip=True)
        ref_bounds = ref.shape[:2][::-1]
        ref_mesh = get_face_mesh(ref, visualise=False)
        ref_rot = get_mesh_x_rotation(ref_mesh, ref_bounds)

    print(f"  Face Orientations (r, c): {ref_rot, content_rot}")

    ref_mesh_vis = visualize_landmarks(ref, ref_mesh)
    vis_mesh_faces(content, content_mesh, show=False)

    out = np.zeros_like(content)
    transform(seg, ref_mesh, content_mesh, out)

    if show:
        cv2.imshow("transformed", out)
        cv2.imshow("original", seg)
        cv2.waitKey(0)

    return content, out, style, style_sem

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Aricimboldo ST")
    parser.add_argument('-t',   '--type',         type=str, required=True, 
                        choices=['vertumnus', 'summer', 'floral', 'four_seasons'])
    parser.add_argument('-c',   '--content',      type=str, required=True)
    parser.add_argument('-d',   '--destination',  type=str, required=True)
    parser.add_argument('--mirror',               action='store_true')
    parser.add_argument('--no-flip',              action='store_true')

    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    content, content_sem, style, style_sem = main(
        args.type,
        args.content,
        args.no_flip,
        show=False
    )

    output_pth = os.path.join(args.destination, f"{run_timestamp}")

    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    print(f"Exporting to [{output_pth}]")
    cv2.imwrite(os.path.join(output_pth, "content.png"), content)
    cv2.imwrite(os.path.join(output_pth, "content_sem.png"), content_sem)
    cv2.imwrite(os.path.join(output_pth, "style.png"), style)
    cv2.imwrite(os.path.join(output_pth, "style_sem.png"), style_sem)