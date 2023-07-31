import argparse
import os
import numpy as np
from PIL import Image
from datetime import datetime
import cv2
import warnings

import tensorflow as tf
from keras.applications import vgg19
from keras import models
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import MaxPooling2D, InputLayer

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

### HYPERPARAMS

# VGG Layers used to extract content and style representations.
LATENT_CONTENT_LAYERS = ['block5_conv2']
LATENT_STYLE_LAYERS   = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
N_CONTENT_LAYERS = len(LATENT_CONTENT_LAYERS)
N_STYLE_LAYERS   = len(LATENT_STYLE_LAYERS)

# VGG Preprocessing normalization means
IMAGENET_MEANS = np.array([
    103.939, 
    116.779, 
    123.68
])
MIN_VGG_V = -IMAGENET_MEANS
MAX_VGG_V = 255-IMAGENET_MEANS

MAX_FACE_IDX = 30000

def get_face(face_path, idx=None):
    if not idx is None:
        if idx >= MAX_FACE_IDX:
            raise ValueError(f"ERROR: Face idx out of range (MAX={MAX_FACE_IDX})")
    else:
        idx = np.random.randint(0, MAX_FACE_IDX)

    path = f"{face_path}\CelebA-HQ-img\{idx}.jpg"

    combined_masks = None

    return path, combined_masks

def load_seg(path, width, height=None):
    im = cv2.imread(path)

    scale = width / im.shape[1] # This has to be the most stupid convention ever.

    width = int(width)
    newsize = (width, round(im.shape[0]*scale)) if height is None else (width, height) 
    
    im = cv2.resize(im, newsize, interpolation=cv2.INTER_NEAREST)
    return im

def process_multi_seg(csem, ssem):
    style_cols = np.unique(ssem.reshape(-1, 3), axis=0)
    content_cols = np.unique(csem.reshape(-1, 3), axis=0)

    print(f"N Colours Detected: Content - {len(content_cols)}, Style - {len(style_cols)}")

    mask_stack_s = []
    mask_stack_c = []
    for c in content_cols:
        m_s = np.all(ssem == c, axis=-1).astype(np.float32)
        m_c = np.all(csem == c, axis=-1).astype(np.float32)

        mask_stack_s.append(m_s)
        mask_stack_c.append(m_c)

    return np.stack(mask_stack_c, axis=2), np.stack(mask_stack_s, axis=2)

def process_seg(sem, n_col=0):
    cleaned = None
    colours = None

    if n_col > 0:

        flat_sem = np.float32(sem.reshape((-1, 3)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(flat_sem, n_col, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)

        cleaned = centers[labels.flatten()].reshape(sem.shape)
        colours = np.unique(cleaned.reshape(-1,3), axis=0)

    elif n_col < 1:
        # Detect number of colours automatically. TODO: Is this even needed?
        colours = np.unique(sem.reshape(-1,3), axis=0)
        n_col = len(colours)
        print(f"Detected {n_col} colours in segmentation.")

        REASONABLE_N_COL = 30
        if n_col > REASONABLE_N_COL:
            warnings.warn(f"""Warning... Number of detected colours ({n_col}) in semantic segmentation exceeds a reasonable amount ({REASONABLE_N_COL}). If this is unexpected, please remove intermediary colors (such as those from the result of anti-aliasing), or specify the parameter `ncol`. The program may run out of memory.""")
        elif n_col < 2:
            warnings.warn(f"No segmentation detected in mask img. The mask was a solid colour (n={n_col}); spacial control may not have intended results.")

        cleaned = sem
    
    mask_stack = []
    for c in colours:
        m = np.all(cleaned == c, axis=-1).astype(np.float32)
        mask_stack.append(m)

    return np.stack(mask_stack, axis=2), cleaned, n_col

def feature_extraction_model():
    """Style and content extraction model using VGG19 as a basis."""
    vgg = vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    # vgg.summary()

    c_outputs = [vgg.get_layer(layer_name).output for layer_name in LATENT_CONTENT_LAYERS]
    s_outputs = [vgg.get_layer(layer_name).output for layer_name in LATENT_STYLE_LAYERS]
    combined_out = c_outputs + s_outputs

    return models.Model(vgg.input, combined_out)

def downscale_model(vgg: models.Model, input_shape):
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    outputs = [model.input]

    for layer in vgg.layers:
        copied_layer = None

        if "pool" in layer.name:
            copied_layer = MaxPooling2D.from_config(layer.get_config())

        if not copied_layer is None:
            model.add(copied_layer)
            outputs.append(copied_layer.output)

    func_model = models.Model(model.input, outputs)
    # func_model.summary()

    return func_model

def get_guidance_channels(vgg, sem_masks):
    masks_in = np.expand_dims(sem_masks, axis=0)

    d_model = downscale_model(vgg, sem_masks.shape)
    
    layer_masks = d_model(masks_in)
    output = []
    for m in layer_masks:
        o = np.squeeze(np.array(m), axis=0)
        o = o.transpose([2, 0, 1]) # [w, h, n] -> [n, w, h]
        output.append(o)

    return output

def load_img(path, width, height=None):
    """Loads target image from path into np array"""
    img = Image.open(path).convert('RGBA')
    alpha_blend = Image.new('RGBA', img.size, (255,255,255)) # Composite potential alpha channels with white
    img = Image.alpha_composite(alpha_blend, img).convert('RGB')

    width = int(width)
    scale = width / img.width

    newsize =  (width, round(img.size[1]*scale)) if height is None else (width, height)
    
    img = img.resize(newsize, Image.LANCZOS)
    img = tf.keras.utils.img_to_array(img)
    
    # Add empty dim for batch input later
    return np.expand_dims(img, axis=0)

def load_and_preprocess(path, width, height=None):
    """Loads target image from path and preprocesses it so that in can be passed into the feature extraction model (VGG19)"""
    img = load_img(path, width, height)

    img = vgg19.preprocess_input(img)
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

def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def gram_matrix(f):
    # TODO: Could use full normalisation term here.
    channels = int(f.shape[-1])
    a = tf.reshape(f, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

@tf.function
def guided_gram_matrix(f, mf):

    channels = int(f.shape[-1])
    a = tf.reshape(f, [channels, -1])
    n = tf.shape(a)[0]

    R = mf.shape[0]
    M = mf.shape[1] * mf.shape[2]
    mx = mf.reshape([R, M])

    guided_gram = []
    for i in range(R):
        mask = mx[i]
        # This uses a lot of memory... too bad!
        masked_a = a * mask
        gram = tf.matmul(masked_a, masked_a, transpose_a=True) / tf.cast(n, tf.float32)
        del masked_a
        guided_gram.append(gram)
        del gram

    return tf.stack(guided_gram)

def get_content_style_features(model, content, style):
    
    f_content = model(content)
    f_content = [layer[0] for layer in f_content[:N_CONTENT_LAYERS]]

    f_style   = model(style)
    f_style   = [layer[0] for layer in f_style[N_CONTENT_LAYERS:]] # Apply gram mat to style features

    return f_content, f_style

@tf.function
def compute_content_loss(vp, v):
    """MSE + Normalisation"""
    return tf.reduce_mean(tf.square(vp - v))

@tf.function
def compute_style_loss(vp, v):
    """MSE between gram-matrix style features"""
    return tf.reduce_mean(tf.square(gram_matrix(vp) - gram_matrix(v)))

@tf.function
def compute_guided_style_loss(vp, v, mp, m):
    return tf.reduce_mean(tf.square(guided_gram_matrix(vp, mp) - guided_gram_matrix(v, m)))

@tf.function
def compute_tv_loss(img):
    dx = img[:, :, 1:, :] - img[:, :, :-1, :]
    dy = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(dx**2) + tf.reduce_mean(dy**2)
    
@tf.function
def compute_loss(model, content_weight, style_weight, tv_weight, tv_factor, result, f_content, f_style, f_content_m, f_style_m):
    new_features = model(result)
    new_f_content = new_features[:N_CONTENT_LAYERS]
    new_f_style   = new_features[N_CONTENT_LAYERS:]

    c_loss, s_loss, tv_loss = 0, 0, 0

    # Average content loss from layers
    c_layer_weight = 1.0 / float(N_CONTENT_LAYERS)
    for f, new_f in zip(f_content, new_f_content):
        c_loss += c_layer_weight * compute_content_loss(new_f[0], f)

    # Average style loss from layers
    s_layer_weight = 1.0 / float(N_STYLE_LAYERS)
    for f, new_f, style_m, content_m in zip(f_style, new_f_style, f_style_m, f_content_m):
        s_loss += s_layer_weight * compute_guided_style_loss(new_f[0], f, style_m, content_m)

    tv_loss += compute_tv_loss(result)

    loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tf.pow(tv_loss, tv_factor)
    return loss, c_loss, s_loss, tv_loss

@tf.function
def compute_gradients(gradient_args):
    with tf.GradientTape() as tape:
        losses = compute_loss(**gradient_args)

    loss = tape.gradient(losses[0], gradient_args['result'])

    return loss, losses

@tf.function
def optimize(opt, gradient_args):
    result = gradient_args['result']
    gradients, losses = compute_gradients(gradient_args)
    opt.apply_gradients([(gradients, result)])
    clipped = tf.clip_by_value(result, MIN_VGG_V, MAX_VGG_V)
    result.assign(clipped)
    return losses

def spatial_sst(content, style, csem, ssem, initial_img, ncol, content_weight, style_weight, tv_weight, tv_factor, n_iterations, learning_rate, verbose, visualize, interval):

    # Do this semi-heavy computation before tf eats all the memory.

    c_masks, s_masks = process_multi_seg(csem, ssem,)

    if verbose:
        print("Processed guidance masks.")

    model = feature_extraction_model()
    if (verbose): 
        print("Loaded Model VGG19")

    f_content, f_style = get_content_style_features(model, content, style)
    f_content_m = get_guidance_channels(model, c_masks)
    f_style_m = get_guidance_channels(model, s_masks)
    if (verbose):
        print("Extracted content and style features", end="\n\n")

    training_t_start = datetime.now()
    
    # Start with content image, and 'optimise' to incorporate style
    if initial_img is None:
        result = tf.Variable(content, dtype=tf.float32)
    else:
        result = tf.Variable(initial_img, dtype=tf.float32)

    # Optimiser
    opt = Adam(learning_rate=learning_rate, epsilon=1e-1)
    gradient_args = {
        'model': model,
        'content_weight': content_weight,
        'style_weight': style_weight,
        'tv_weight': tv_weight,
        'tv_factor': tv_factor,
        'result': result,
        'f_content': f_content,
        'f_style': f_style,
        'f_content_m': f_content_m,
        'f_style_m': f_style_m
    }

    best_result = None
    best_loss = float('inf')
    iter = 0
    current_iterations = n_iterations

    try: 
        while current_iterations > 0:
            for i in range(current_iterations):
                iter_t_start = datetime.now()

                print(f"Running iter {i}")

                losses = optimize(opt, gradient_args)
                training_loss, c_loss, s_loss, tv_loss = losses

                if training_loss < best_loss:
                    best_loss = training_loss
                    best_result = un_preprocess_img(result.numpy())

                if (visualize and i % interval == 0):
                    cv2.imshow("Style Transfer Preview", cv2.cvtColor(un_preprocess_img(result.numpy()), cv2.COLOR_RGB2BGR))
                    cv2.waitKey(50)

                dt = datetime.now() - iter_t_start
                elapsed_ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0

                iter += 1
                if (i % interval == 0):
                    print (f"[iter {i}] (t={elapsed_ms:.0f}ms) | Loss: {training_loss:.3e} | C Loss: {c_loss:.3e} | S Loss: {s_loss:.3e} | TV Loss: {tv_loss:.3e}")

            new_iters = input(f"\nContinue? (Number of iterations [Y] / Enter [N]): ")
            print("")
            current_iterations = int(new_iters) if (new_iters.isnumeric() and int(new_iters) > 0) else 0

    except KeyboardInterrupt:
        print(f"Exited Early (i = {iter})")

    training_dt = (datetime.now() - training_t_start).total_seconds()
    hours = divmod(training_dt, 3600)
    minutes = divmod(hours[1], 60)
    seconds = divmod(minutes[1], 1)

    print(f"Finished ({hours[0]}h {minutes[0]}m {seconds[0]}s). Final Loss: {best_loss:.3e}")

    return best_result, best_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Spacial Control In Style Transfer (Insp. Gatys 2017)")
    parser.add_argument('-c',   '--content',        type=str, required=True)
    parser.add_argument('-csem','--content-sem',    type=str, required=True)
    parser.add_argument('-s',   '--style',          type=str, required=True)
    parser.add_argument('-ssem','--style-sem',      type=str, required=True)
    parser.add_argument('-ii',  '--initial-img',    type=str, default=None)
    parser.add_argument('-ncol','--n-colors',       type=int, default=0)
    parser.add_argument('-d',   '--destination',    type=str, required=True)

    parser.add_argument('--celeb-dataset-dir',      type=str, default="S:\Dissertation Dataset\CelebAMask-HQ")
    
    parser.add_argument('-n',   '--num-iter',       type=int,   default=10)
    parser.add_argument('-lr',  '--learning-rate',  type=float, default=10)
    parser.add_argument('-iw',  '--image-width',    type=float, default=512)

    parser.add_argument('-cw',  '--content-weight', type=float, default=1)
    parser.add_argument('-sw',  '--style-weight',   type=float, default=0.2)
    parser.add_argument('-tvw', '--tv-weight',      type=float, default=8.5e-7)
    parser.add_argument('-tvf', '--tv-factor',      type=float, default=1.0)

    parser.add_argument('-v',   '--verbose',        action='store_true')
    parser.add_argument('-vis', '--visualize',      action='store_true')
    parser.add_argument('-int', '--interval',       type=int, default=1)
    parser.add_argument('-gpu', '--gpu-idx',        type=int, default=-1)

    parser.add_argument('-ns',  '--no-standalone',  action='store_true')
    parser.add_argument('-nc',  '--no-compiled',    action='store_true')

    args = parser.parse_args()


    # GPU Configuration
    devices = tf.config.experimental.list_physical_devices('GPU')
    device  = None if args.gpu_idx == -1 else devices[args.gpu_idx]

    if device is None:
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) < 1:
            device = tf.config.list_physical_devices('CPU')[0]
        else:
            device = devices[0]
    elif device.device_type == 'GPU':
        try:
            tf.config.set_visible_devices(device, 'GPU')
            tf.config.experimental.set_memory_growth(device, True)

        except RuntimeError as e:
            print(e)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load imgs for NST

    # If content is an idx, load from celeb-mask-ahq dataset.
    # This is a hack, but the broadcasting isn't working well so just make the style and content images the same size...
    if args.content.isnumeric():
        idx = int(args.content)
        content_path, mask = get_face(args.celeb_dataset_dir) if idx < 0 else get_face(args.celeb_dataset_dir, idx)
        content_img = load_and_preprocess(content_path, args.image_width)
    else: 
        content_img = load_and_preprocess(args.content, args.image_width)

    style_img = load_and_preprocess(args.style, args.image_width, content_img.shape[1])

    if args.initial_img:
        initial_img = load_and_preprocess(args.initial_img, args.image_width, content_img.shape[1])
    else:
        initial_img = None

    if (args.verbose):
        print("Loaded Source Images (Content, Style)")

    # Load Semantic Segmentations
    content_sem = load_seg(args.content_sem, args.image_width)
    style_sem = load_seg(args.style_sem, args.image_width, content_sem.shape[0])
    
    if (args.verbose):
        print("Loaded Segementations (CSem, SSem)")
    
    # Do SST optimization
    result, loss = spatial_sst(
        content_img, 
        style_img,
        content_sem,
        style_sem,
        initial_img,
        args.n_colors,
        args.content_weight,
        args.style_weight,
        args.tv_weight,
        args.tv_factor,
        args.num_iter,
        args.learning_rate,
        args.verbose,
        args.visualize,
        args.interval
    )

    export_path = args.destination + "/gatys2017/"
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Standalone
    if not (args.no_standalone):
        im = Image.fromarray(result)
        im.save(export_path + f"{run_timestamp}.png")
        if (args.verbose):
            print(f"Standalone image exported to [{export_path}{run_timestamp}.png")

    # Compiled
    if not (args.no_compiled):
        ims = [Image.fromarray(i) for i in [un_preprocess_img(content_img), un_preprocess_img(style_img), result]]
        get_concat_h_multi_resize(ims).save(export_path + f"{run_timestamp}_compiled.png")
        if (args.verbose):
            print(f"Compiled image exported to [{export_path}{run_timestamp}_compiled.png")
