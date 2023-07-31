from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import torch
import numpy as np
from enum import Enum
from datetime import datetime
from sklearn.cluster import AffinityPropagation
import os
import argparse

# Automatic segmentation script and prompting interface for SAM.

class PromptMode(Enum):
    POSITIVE = 1
    NEGATIVE = 2
    BOX = 3
    ALL = 4

def load_sam(model, ckpt_path):
    print(f"  [Models can use CUDA: {torch.cuda.is_available()}]")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # TODO: I'm not sure if cpu is a valid device here

    sam = sam_model_registry[model](checkpoint=ckpt_path)
    sam.to(device=DEVICE)

    return sam

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# Useful thread explaining why using average hue is not representative.
# https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
def get_dominant_colour(segment, n=3):
    pixels = np.float32(segment.reshape(-1, 3))

    n_colors = n
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant_col = palette[np.argmax(counts)] # Don't include black mask regions
    if np.all(dominant_col == np.array([0, 0, 0]), axis=-1):
        dominant_col = palette[np.argsort(counts)[-2]] # Secondmost dominant colour

    return dominant_col

def affinity_propagation(img, segmentation, use_hist=False, visualize=False):
    # Affinity propagation of mask shapes based on area alone.
    #   TODO: Other statistics that might be useful
    #       Some sort of coherence measure (e.g. how much of a blob each mask is?)
    #       Colour summary statistics (e.g. average / histogram)
    #       Width and height statistics (Would have to rotate the segment to match longest axis: maybe PCA?)

    colours, areas = np.unique(segmentation.reshape(-1, 3), axis=0, return_counts=True)

    max_area = max(areas)

    # Black should always be at index 0 here... If there is errors or unexpected segmentation of background
    #   Then it is probably due to this assumption, could be changed to search and remove.
    if np.all(colours[0] == np.array([0, 0, 0])):
        colours = colours[1:]
        areas = areas[1:]

    print(f"Unique Colours (before clustering): {len(colours)}")

    img_channels = cv2.split(img)

    features = []
    for i, (c, area) in enumerate(zip(colours, areas)):
        mask = np.all(segmentation == c, axis=-1).astype(np.uint8)
        masked = img * mask[..., None]

        # Normalise area to ensure equal weighting to colour info
        seg_features = [area / max_area]

        if use_hist:
            # Use full colour representation
            np.unique(masked.reshape(-1, 3), axis=0, return_counts=True)

            for channel in img_channels:
                hist = cv2.calcHist([channel], [0], mask, [256], [0, 256]) / 255
                seg_features.extend(list(hist))
        # else:
        #     # Use dominant colour
        #     dom_col = get_dominant_colour(masked) / 255
        #     seg_features.extend(dom_col)

        features.append(seg_features)

    features = np.array(features, dtype=object)

    print("Performing Affinity Propagation...")
    clustering = AffinityPropagation(max_iter=2000).fit(features)

    new_colours = clustering.cluster_centers_
    labels = clustering.labels_

    print(f"Unique Colours (after clustering): {len(new_colours)}")

    new_seg = np.copy(segmentation)
    for i, l in enumerate(labels):
        colour_before = colours[i]

        if np.all(colour_before == np.array([0, 0, 0])) \
                or np.all(colour_before == np.array([255, 255, 255])):
            continue
            
        colour_after = colours[l]
        new_seg[np.all(new_seg == colour_before, axis=-1)] = colour_after

        # print(f"Color: {colours[i]}, New Colour: {colours[l]}")

    if visualize:
        cv2.imshow('base', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow('before', segmentation)
        cv2.imshow('after', new_seg)
        cv2.waitKey(0)

    return new_seg

def show_mask(mask, colour_id=-1):

    if colour_id >= 0:

        # Reserve 0 for empty (not drawn) or red (drawn).
        black = np.array([0, 0, 0])
        red = np.array([1, 1, 1])

        if colour_id == 0:
            color = red
        else:
            prng = np.random.RandomState(50 + colour_id)
            color = prng.random(3)
            # Ensure not using the same colour as background
            while np.all(color == black) or np.all(color == red):
                color = prng.random(3)

    else:
        color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape[-2:]

    colored_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    colored_mask = (colored_mask * 255).astype(np.uint8)

    return colored_mask

def draw_prompt_points(img, pts, labels):
    for pt, label in zip(pts, labels):
        col = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, center=pt, radius=5, color=col, thickness=-1)

    return img

def cleanup_masks(im, min_area, colour_background=False):
    colours = np.unique(im.reshape(-1,3), axis=0)
    n_col = len(colours)

    mask_stack = []
    for c in colours:

        if np.all(c == np.array([0, 0, 0])):
            if not colour_background:
                continue

        # TODO: It may be worth doing a morphological close here to remove erroneous pixels, 
        #           now noise, from overlapping masks... Though it might be determental on mask outline quality.
        m = np.all(im == c, axis=-1).astype(np.float32)
        area = np.count_nonzero(m)
        if area > min_area:
            mask_stack.append(m)

    return draw_masks(np.zeros_like(im), mask_stack, 0.0, 1.0)

def split_and_write_segmentation(segmentation, output_dir):
    colours = np.unique(segmentation.reshape(-1,3), axis=0)
    n_col = len(colours)

    seg_dir = os.path.join(output_dir, "segmentation")

    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    for i, c in enumerate(colours):

        if np.all(c == np.array([0, 0, 0])):
            continue

        # TODO: It may be worth doing a morphological close here to remove erroneous pixels, 
        #           now noise, from overlapping masks... Though it might be determental on mask outline quality.
        m = np.all(segmentation == c, axis=-1).astype(np.float32) * 255
        cv2.imwrite(os.path.join(seg_dir, f"{i}.png"), m)

def draw_masks(im, masks, im_weight=0.6, mask_weight=0.4):
    if masks is None or len(masks) == 0:
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    masked = np.zeros_like(im)
    for i, mask in enumerate(masks):

        # Deal with masks from auto generation
        if isinstance(mask, dict):
            mask = mask['segmentation']

        masked = masked + show_mask(mask, colour_id=i)
    mask_image = cv2.addWeighted(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), im_weight, masked, mask_weight, 0.0)

    if masks is None or len(masks) == 0:
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    return mask_image

def render_masks(im, masks, im_weight=0.6, mask_weight=0.4, min_area = 100, colour_background=False):
    mask = draw_masks(im, masks, im_weight, mask_weight)
    mask = cleanup_masks(mask, min_area, colour_background)

    return mask

def await_masks(img, automatic=False, model="vit_h", model_path="src/kirillov2023/checkpoints/sam_vit_h_4b8939.pth"):

    print("Loading Segmentation Models...")
    sam = load_sam(model, model_path)
    # Whole image masking
    all_predictor = SamAutomaticMaskGenerator(sam,
        min_mask_region_area=100 # TODO: This should be adjusted based on img resolution potentially?
    )

    if automatic:
        print("Generating Auto Masks")
        masks = all_predictor.generate(img)
        print("Generated.")
        return masks

    # Point based
    point_predictor = SamPredictor(sam)
    point_predictor.set_image(img)

    logits = None
    prompt_masks = []
    all_masks = None
    masks = None
    masks_inc_hover = None
    need_mask_redraw = False
    
    show_hover = True

    prompt_mode = PromptMode.POSITIVE

    mouse_pos = [-1, -1]
    input_points = []
    input_labels = np.array([])
    
    def adjust_masks(e, x, y, flags, params):
        nonlocal mouse_pos, input_points, input_labels, \
                    img, point_predictor, masks, logits, need_mask_redraw, prompt_mode

        if e == cv2.EVENT_MOUSEMOVE:
            mouse_pos = [x, y]

        elif e == cv2.EVENT_LBUTTONDOWN or e == cv2.EVENT_RBUTTONDOWN:
            input_points.append([x, y])
            label = 0 if e == cv2.EVENT_RBUTTONDOWN else 1
            if prompt_mode == PromptMode.NEGATIVE: label = 1 - label
            input_labels = np.append(input_labels, label)
            masks, scores, logits = point_predictor.predict(
                point_coords=np.array(input_points),
                point_labels=input_labels,
                mask_input=None if logits is None else logits[None, :, :],
                multimask_output=False
            )
            logits = logits[np.argmax(scores), :, :]
            need_mask_redraw = True

    cv2.namedWindow('Masks')
    cv2.setWindowProperty('Masks', cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback('Masks', adjust_masks)
    render = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    while True:
        if show_hover and (prompt_mode == PromptMode.POSITIVE or prompt_mode == PromptMode.NEGATIVE):
            # Show hover
            if mouse_pos[0] >= 0 and mouse_pos[1] >= 0 and mouse_pos[0] < img.shape[0] and mouse_pos[1] < img.shape[1]:
                hover_label = 1 if prompt_mode == PromptMode.POSITIVE else 0
                masks_inc_hover, _, _ = point_predictor.predict(
                    point_coords=np.array(input_points + [mouse_pos]),
                    point_labels=np.append(input_labels, np.array([hover_label])),
                    mask_input=None if logits is None else logits[None, :, :],
                    multimask_output=False
                )
                render = draw_masks(img, masks_inc_hover)
                render = draw_prompt_points(render, input_points, input_labels) # This is gross move to render
                need_mask_redraw = False

        elif need_mask_redraw: 
            render = draw_masks(img, masks)
            render = draw_prompt_points(render, input_points, input_labels)
            need_mask_redraw = False

        cv2.imshow("Masks", render)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            prompt_mode = PromptMode.ALL
            print("Generating Auto Masks...")
            all_masks = all_predictor.generate(img)
            need_mask_redraw = True
            print("Generated.")
            break
        elif k == ord('d'):
            prompt_masks.extend(masks)
            masks = None
            input_points = []
            input_labels = np.array([])
            need_mask_redraw=True
        elif k == ord('p'):
            print("Switched to POSITIVE prompt mode")
            prompt_mode = PromptMode.POSITIVE
            need_mask_redraw = True
        elif k == ord('n'):
            print("Switched to NEGATIVE prompt mode")
            prompt_mode = PromptMode.NEGATIVE
            need_mask_redraw = True
        elif k == ord('h'):
            print(f"Show hover set to {str(show_hover).upper()}")
            show_hover = not show_hover
            need_mask_redraw = True
        elif k == ord('q'):
            print("Finalised Masks.")
            break
    
    cv2.destroyWindow('Masks')
    
    m = prompt_masks + all_masks

    return m

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Segment Anything Interface")
    parser.add_argument('-i',   '--image',        type=str, required=True)
    parser.add_argument('-d',   '--destination',  type=str, required=True)
    parser.add_argument('-m',   '--model',        type=str, default='vit_h', choices=['default', 'vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('-mp',  '--model-path',   type=str, default="src/kirillov2023/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument('--omit-bg',              action='store_true')
    parser.add_argument('--use-hist',             action='store_true')
    parser.add_argument('--render-individual',    action='store_true')

    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    img = load_img(args.image)

    print("[GENERATING SEGMENTS]", end="\n\n")
    masks = await_masks(img, automatic=True, model=args.model, model_path=args.model_path)
    print("")

    # masked_img = render_masks(img, masks)
    masked_img = render_masks(np.zeros_like(img), masks, colour_background=(not args.omit_bg))

    print("[CLUSTERING SEGMENTS]", end="\n\n")
    clustered_mask = affinity_propagation(img, masked_img, use_hist=args.use_hist, visualize=False)
    print("")

    out_folder = os.path.join(args.destination, f"{run_timestamp}/")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    cv2.imwrite(os.path.join(out_folder, f"full.png"), masked_img)
    cv2.imwrite(os.path.join(out_folder, f"clustered_mask.png"), clustered_mask)

    if args.render_individual:
        split_and_write_segmentation(clustered_mask, out_folder)

