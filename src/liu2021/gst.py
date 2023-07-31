import argparse
import os
import matlab.engine
import numpy as np
from PIL import Image
from datetime import datetime
import cv2

# Python Interface to Liu et al (2021) MATLAB Geo Warping Code.

MAX_FACE_IDX = 30000

def get_face(face_path, idx=None):
    label_list = [
        'skin',     'nose',     'eye_g',    'l_eye',    'r_eye',    'l_brow', 
        'r_brow',   'l_ear',    'r_ear',    'mouth',    'u_lip',    'l_lip', 
        'hair',     'hat',      'ear_r',    'neck_l',   'neck',     'cloth'
    ]
    color_list = [
        [  0,   0,   0], 
        [204,   0,   0], [ 76, 153,   0], [204, 204,   0], [ 51,  51, 255], [204,   0, 204], [  0, 255, 255], 
        [255, 204, 204], [102,  51,   0], [255,   0,   0], [102, 204,   0], [255, 255,   0], [  0,   0, 153], 
        [  0,   0, 204], [255,  51, 153], [  0, 204, 204], [  0,  51,   0], [255, 153,  51], [  0, 204,   0]
    ]

    if not idx is None:
        if idx >= MAX_FACE_IDX:
            raise ValueError(f"ERROR: Face idx out of range (MAX={MAX_FACE_IDX})")
    else:
        idx = np.random.randint(0, MAX_FACE_IDX)

    path = f"{face_path}\CelebA-HQ-img\{idx}.jpg"

    mask_folder_num = idx // 2000
    combined_masks = np.zeros((512, 512, 3))
    for i, label in enumerate(label_list):
        filename = os.path.join(f"{face_path}\CelebAMask-HQ-mask-anno", str(mask_folder_num), str(idx).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            im = cv2.imread(filename)
            im = im[:, :, 0]
            combined_masks[im != 0] = color_list[i +  1]

    return path, combined_masks.astype(np.uint8)

def load_img(path, width):
    """Loads target image from path into np array"""
    img = Image.open(path).convert('RGBA')
    alpha_blend = Image.new('RGBA', img.size, (255,255,255)) # Composite potential alpha channels with white
    img = Image.alpha_composite(alpha_blend, img).convert('RGB')

    scale = width / img.width
    
    img = img.resize((width, round(img.height*scale)), Image.LANCZOS)
    
    return img

def un_preprocess_img(processed_img):
    img = processed_img.copy()
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0) # Remove batch dim

    assert len(img.shape) == 3, ("Expected image from batch process with shape (1, w, h, c), but got (w, h, c)")

    # Re-add means from VGG preprocessing (un-normalize)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] # Reverse

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def warp_image(img, flow, inverse=False, interpolation=None):
    mapx_base, mapy_base = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    if inverse:
        mapx = mapx_base - flow[:, :, 0]
        mapy = mapy_base - flow[:, :, 1]
    else:
        mapx = mapx_base + flow[:, :, 0]
        mapy = mapy_base + flow[:, :, 1]

    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    warped = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=interpolation)

    return warped

def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC, use_min=True):
    
    if use_min:
        height_thresh = min(im.height for im in im_list)
    else:
        height_thresh = max(im.height for im in im_list)

    im_list_resize = [im.resize((int(im.width * height_thresh / im.height), height_thresh),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, height_thresh))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def get_flow(content_path, style_path, image_width):
    print("Running Geometric Warping...")
    eng = matlab.engine.start_matlab()
    try:
        eng.cd('src/liu2021/warping/geometric_warping')
        warp, flow, flow_colour = eng.geo_warping(content_path, style_path, image_width, nargout=3) # this has to be the single most annoying 
        warp, flow, flow_colour = np.asarray(warp), np.asarray(flow), np.asarray(flow_colour)

        eng.quit()

    except Exception as e:
        eng.quit()
        print(e)
        raise ValueError("ERROR: Matlab script failed. See error above for details.")

    print("Geometric Warping Finished")

    return flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Gatys 2016 Neural Style Transfer")
    parser.add_argument('-c',   '--content',        type=str, required=True)
    parser.add_argument('-s',   '--style',          type=str, required=True)
    parser.add_argument('-d',   '--destination',    type=str, required=True)
    
    parser.add_argument('-iw',  '--image-width',    type=int, default=512)

    parser.add_argument('-v',   '--verbose',        action='store_true')
    parser.add_argument('-vis', '--visualize',      action='store_true')

    parser.add_argument('-ns',  '--no-standalone',  action='store_true')
    parser.add_argument('-nc',  '--no-compiled',    action='store_true')
    parser.add_argument('--celeb-dataset-dir',      type=str, default="S:\Dissertation Dataset\CelebAMask-HQ")

    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.verbose:
        print("Running Geometric Warping...")

    # Get Face Path
    content_path = args.content
    mask = np.zeros((args.image_width, args.image_width, 3)).astype(np.uint8)
    if args.content.isnumeric():
        idx = int(args.content)
        content_path, mask = get_face(args.celeb_dataset_dir) if idx < 0 else get_face(args.celeb_dataset_dir, idx)

    content_path = os.path.abspath(content_path)
    style_path = os.path.abspath(args.style)

    ### Geometric Warping
    eng = matlab.engine.start_matlab()
    try:
        eng.cd('src/liu2021/warping/geometric_warping')
        warp, flow, flow_colour = eng.geo_warping(content_path, style_path, args.image_width, nargout=3) # this has to be the single most annoying 
        warp, flow, flow_colour = np.asarray(warp), np.asarray(flow), np.asarray(flow_colour)

        eng.quit()

    except Exception as e:
        eng.quit()
        print(e)
        raise ValueError("ERROR: Matlab script failed. See error above for details.")

    print("Geometric Warping Finished")

    # Load Content-style images for later use
    content_img = load_img(content_path, args.image_width)
    style_img   = load_img(style_path, args.image_width)

    # Warp mask
    mask = Image.fromarray(mask)
    mask = mask.resize((256, 256), Image.BILINEAR)
    mask = np.array(mask)
    warped_mask = warp_image(mask, flow)

    # Resize to fit style image
    warped_mask = Image.fromarray(warped_mask)
    warped_mask = warped_mask.resize(style_img.size, Image.BILINEAR)

    # Overlay with style image
    overlay = Image.blend(warped_mask, style_img, alpha=0.65)

    export_path = os.path.join(args.destination, "liu2021/")
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Standalone
    if not (args.no_standalone):

        for n, i in [
            ("raw", load_img(content_path, args.image_width)), 
            ("mask", Image.fromarray(mask)),
            ("warped", Image.fromarray(warp)),
            ("warped_mask", warped_mask)
        ]:
            i.save(export_path + f"{run_timestamp}_{n}.png")
            if (args.verbose):
                print(f"Standalone image exported to [{export_path}{run_timestamp}_{n}.png]")

    # Warping
    if not (args.no_compiled):
        ims = [load_img(content_path, args.image_width), Image.fromarray(mask), load_img(style_path, args.image_width), Image.fromarray(warp), overlay, Image.fromarray(flow_colour)]
        get_concat_h_multi_resize(ims).save(export_path + f"{run_timestamp}_warp.png")
        if (args.verbose):
            print(f"Compiled Warping image exported to [{export_path}{run_timestamp}_warp.png]")