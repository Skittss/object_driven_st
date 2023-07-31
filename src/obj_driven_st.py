from liu2021.gst import get_flow, warp_image, load_img
from kirillov2023.segmentation import await_masks, render_masks, affinity_propagation
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import argparse

# Automatic ODST pipeline

# TODO: Unused
def create_collage(content, style, masks, flow):

    # Create an arcimboldo-style collage

    masked = np.zeros_like(content)
    upscale_x, upscale_y = 256 / style.shape[0], 256 / style.shape[1]
    for mask in masks:

        # Deal with masks from auto generation
        if isinstance(mask, dict):
            mask = mask['segmentation']

        mass_y, mass_x = np.where(mask == True)
        cx = np.average(mass_x); cy = np.average(mass_y)
        cx = int(cx * upscale_x); cy = int(cy * upscale_y)
        shift = warp_pt(cx, cy, flow, inverse=True)

        masked_style = translate_img(style * mask[..., None], *shift)
        np.copyto(masked, masked_style, where=masked_style > 0)

    cv2.imshow('masked', cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
        
def warp_pt(x, y, flow, inverse=False):
    
    flow_coeff = -1 if inverse else 1

    warp_x = flow_coeff * flow[x, y, 0]
    warp_y = flow_coeff * flow[x, y, 1]

    return (int(warp_x), int(warp_y))

def translate_img(image, tx, ty):
    N, M = image.shape[:2]

    image_translated = np.zeros_like(image)
    image_translated[max(tx,0):M+min(tx,0), max(ty,0):N+min(ty,0), :] = image[-min(tx,0):M-max(tx,0), -min(ty,0):N-max(ty,0), :]  

    return image_translated

def warp_and_resize_masks(masks, size, flow, inverse=False):

    resized = []
    for mask in masks:
        # Deal with masks from auto generation
        if isinstance(mask, dict):
            mask = mask['segmentation']

        mask = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = warp_image(mask.astype(np.float32), flow, inverse=inverse, interpolation=cv2.INTER_NEAREST)

        resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        resized.append(resized_mask)
        
    return resized

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Object Driven Style Transfer")
    parser.add_argument('-c',   '--content',        type=str, required=True)
    parser.add_argument('-s',   '--style',          type=str, required=True)
    parser.add_argument('-d',   '--destination',    type=str, required=True)
    parser.add_argument('--omit-bg',                action='store_true')
    parser.add_argument('--cluster',                action='store_true')
    parser.add_argument('--prompt-seg',             action='store_true')
    parser.add_argument('--vis',                    action='store_true')

    parser.add_argument('-iw',  '--image-width',    type=int, default=512)

    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load images
    WIDTH = args.image_width
    STYLE_PATH = os.path.abspath(args.style)
    CONTENT_PATH = os.path.abspath(args.content)

    content_img = load_img(CONTENT_PATH, WIDTH)
    HEIGHT = content_img.height
    style_img = np.array(load_img(STYLE_PATH, WIDTH))

    # Do Geo Warping
    print("[GEOMETRIC WARP]", end="\n\n")
    flow = get_flow(CONTENT_PATH, STYLE_PATH, WIDTH)
    print()

    # Do Segmentation
    print("[AUTO SEGMENTATION]", end="\n\n")
    masks = await_masks(style_img, automatic=(not args.prompt_seg), model_path="src/kirillov2023/checkpoints/sam_vit_h_4b8939.pth")
    print()

    # Clustering
    masks_vis = render_masks(np.zeros_like(style_img), masks, im_weight=0, mask_weight=1, colour_background=(not args.omit_bg))
    if args.cluster:
        masks_vis = affinity_propagation(style_img, masks_vis, use_hist=False, visualize=False)
    masks_vis_full_size = Image.fromarray(masks_vis)

    # Object Correspondence Step
    #   Re-drawing the mask produces color inconsistencies, so just warp the image instead.
    unwarped_masks = cv2.resize(masks_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
    unwarped_masks = warp_image(unwarped_masks, flow, inverse=True, interpolation=cv2.INTER_NEAREST)
    if not args.omit_bg:
        unwarped_masks[np.all(unwarped_masks == np.array([0, 0, 0]), axis=-1)] = np.array([255, 255, 255])
    unwarped_masks_vis = cv2.resize(unwarped_masks, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    resized_style = np.array(Image.fromarray(style_img).resize((256,256), Image.BILINEAR))
    unwarped_style = warp_image(resized_style, flow, inverse=True)
    unwarped_style = Image.fromarray(unwarped_style)
    unwarped_style  = unwarped_style.resize((WIDTH, HEIGHT), Image.NEAREST)

    # Visualisation
    ALPHA_BLEND = 0.2
    content_preview = Image.blend(Image.fromarray(unwarped_masks_vis), content_img, alpha=ALPHA_BLEND)
    style_preview = Image.blend(masks_vis_full_size, Image.fromarray(style_img), alpha=ALPHA_BLEND)

    # Export Results (c, csem, s, ssem)
    export_folder = os.path.join(args.destination, f"{run_timestamp}")

    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    cv2.imwrite(f"{export_folder}/content.png", cv2.cvtColor(np.array(content_img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{export_folder}/style.png", cv2.cvtColor(np.array(style_img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{export_folder}/content_sem.png", cv2.cvtColor(np.array(unwarped_masks_vis), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{export_folder}/style_sem.png", cv2.cvtColor(np.array(masks_vis_full_size), cv2.COLOR_RGB2BGR))

    print("Done.")

    # Vis
    if args.vis:
        cv2.imshow('content', cv2.cvtColor(np.array(content_preview), cv2.COLOR_RGB2BGR))
        cv2.imshow('style', cv2.cvtColor(np.array(style_preview), cv2.COLOR_RGB2BGR))
        cv2.imshow('unwarped style', cv2.cvtColor(np.array(unwarped_style), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)