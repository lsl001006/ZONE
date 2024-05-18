import os
from PIL import Image
import numpy as np
import cv2
import torch
from scipy.ndimage import label, find_objects, measurements, binary_fill_holes
from skimage.measure import regionprops
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel
from functools import reduce
from time import time

def draw_mask_to_image(image, mask, alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)
    mask_rgba = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
    overlay = np.zeros_like(image, dtype=np.uint8)

    for c in range(0, 3):
        overlay[:, :, c] = (mask_rgba[:, :, c] * (mask_rgba[:, :, 3] / 255.0) 
                            +  image[:, :, c] * (1.0 - mask_rgba[:, :, 3] / 255.0))
    result = cv2.addWeighted(image, 1, overlay, alpha, 0)

    x,y = cal_max_mask_barycentre(mask)
    cv2.circle(result, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    return result

def extract_connected_regions(mask):
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_mask = np.zeros_like(mask)
    connected_region_images = []
    connected_region_area = []
    for contour in contours:
        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        region_image = cv2.bitwise_and(mask, mask, mask=region_mask)
        if np.sum(region_image>128) < 16*16:
            continue

        connected_region_images.append(region_image)
        connected_region_area.append(np.sum(region_image==255))
        full_mask = cv2.bitwise_or(full_mask, region_mask)
    return connected_region_images, full_mask, connected_region_area

def cal_max_mask_barycentre(mask):
    labeled_mask, num_features = label(mask)
    counts = np.bincount(labeled_mask.flatten())
    max_label = np.argmax(counts[1:]) + 1
    max_mask = labeled_mask == max_label
    props = regionprops(max_mask.astype(np.uint8))
    center_y = int(props[0].centroid[0])
    center_x = int(props[0].centroid[1])

    print("Max connected component centroid: ({:.2f}, {:.2f})".format(center_x, center_y))
    return center_x, center_y

def cal_mask_barycentre(mask, min_area=256):
    labeled_mask, num_features = label(mask)
    props = regionprops(labeled_mask)
    center_coordinates = []

    for prop in props:
        if prop.area >= min_area:
            center_y = int(prop.centroid[0])
            center_x = int(prop.centroid[1])
            center_coordinates.append((center_x, center_y))

    return np.array(center_coordinates)

def get_finemask(mask, sty_image, predictor):
    if isinstance(mask, str):
        mask = np.array(Image.open(mask))
    elif isinstance(mask, Image.Image):
        mask = np.array(mask)
    elif isinstance(mask, np.ndarray):
        pass
    else:
        raise ValueError("Not compatible format for rough mask")
    
    if isinstance(sty_image, str):
        np_img = np.array(Image.open(sty_image))
    elif isinstance(sty_image, Image.Image):
        np_img = np.array(sty_image.convert('RGB'))
    elif isinstance(sty_image, np.ndarray):
        pass
    else:
        raise ValueError("Not compatible format for style image")
    
    coords = cal_mask_barycentre(mask) 
    print(f'Found {len(coords)} connected regions, centers by {coords}.')
    
    predictor = SamPredictor(predictor)
    predictor.set_image(np_img)
    masks, _, _ = predictor.predict(point_coords = coords,point_labels=np.array([1]*len(coords)))
    masks = np.transpose(masks, (1,2,0)).astype(np.uint8)
    bimasks = cv2.cvtColor(masks, cv2.COLOR_RGB2GRAY)
    bimasks[bimasks>0]=255
    np_img[bimasks==0] = 0
    masked_out_pil = Image.fromarray(np_img)
    
    return np_img, bimasks

def dilate_mask(binary_mask, kernel_size = 3, iterations= 10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=iterations)
    dilate_diff = cv2.absdiff(dilated_mask, binary_mask)
    return dilated_mask, dilate_diff

def clip_mask_filter(prompts, image, mask):
    model_id = "/path/to/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    connected_regions, full_mask, connected_region_area = extract_connected_regions(mask)
    max_idx = connected_region_area.index(max(connected_region_area))
    clip_score = []
    for region in connected_regions:
        print(np.sum(region==255))
        region_image = np.expand_dims(region, axis=2)*image
        text = prompts
        inputs = processor(text, images=region_image, return_tensors="pt")
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  
        clip_score.append(logits_per_image)
    
    retain_idxs = []
    for i in range(len(clip_score)):
        if clip_score[i] >= clip_score[max_idx] - 0.5:
            retain_idxs.append(i)
    retain_regions = [connected_regions[j] for j in retain_idxs]
    final_mask = reduce(lambda x,y:x+y, retain_regions)
    
    cv2.imwrite('fm.png', final_mask)
    return final_mask

def filter_small_regions(mask, region_thres=64):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    labeled_image, num_features = label(mask)
    regions = find_objects(labeled_image)
    for i, region in enumerate(regions):
        if region is not None:
            area = measurements.sum(labeled_image[region], labeled_image[region], index=i+1)
            if area < region_thres:
                mask[region] = 0
    return mask

def get_finemask_everything(rough_mask, style_image, predictor):
    if isinstance(style_image, Image.Image):
        style_image = np.array(style_image)
    mask_generator = SamAutomaticMaskGenerator(predictor)
    print('start generating masks...')
    t1 = time()
    segmentations = mask_generator.generate(style_image)
    t2 = time()
    print(t2-t1)
    rough_mask = filter_small_regions(rough_mask)
    fine_mask = find_most_overlapping_segmentation(rough_mask, segmentations)
    fine_mask = filter_small_regions(fine_mask)
    fine_mask[fine_mask>0]=255
    style_image[fine_mask==0] = 0
    masked_out_pil = Image.fromarray(style_image)

    return style_image, fine_mask

def find_most_overlapping_segmentation(mask, segmentations):
    if len(segmentations) == 0:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.max() > 1:
        mask = (mask / mask.max()).astype(np.uint8)
    
    max_intersection_area = 0
    max_iou = 0
    seg_ratio = 0
    best_segmentation = None
    candidate_segs = []
    for object_info in segmentations:
        segmentation = object_info['segmentation']
        intersection = np.logical_and(mask, segmentation)
        intersection_area = np.sum(intersection)
        union = np.logical_or(mask, segmentation)
        union_area = np.sum(union)
        iou = intersection_area / union_area
        if iou > max_iou:
            max_iou = iou
            seg_ratio = intersection_area/np.sum(segmentation.astype(np.uint8))
            best_segmentation = segmentation

    print(f'Best Segmentation IOU: {max_iou:.2f}, seg_ratio:{seg_ratio:.2f}')
    best_segmentation = best_segmentation.astype(np.uint8) * 255
    return best_segmentation

def transfer_color(src, target):
    
    src_b, src_g, src_r = cv2.split(src)
    tgt_b, tgt_g, tgt_r = cv2.split(target)

    (tgt_b_mean, tgt_b_std) = cv2.meanStdDev(tgt_b)
    (tgt_g_mean, tgt_g_std) = cv2.meanStdDev(tgt_g) 
    (tgt_r_mean, tgt_r_std) = cv2.meanStdDev(tgt_r)

    src_b = (src_b - tgt_b_mean) / tgt_b_std
    src_g = (src_g - tgt_g_mean) / tgt_g_std
    src_r = (src_r - tgt_r_mean) / tgt_r_std

    trans_b = src_b * tgt_b_std + tgt_b_mean
    trans_g = src_g * tgt_g_std + tgt_g_mean
    trans_r = src_r * tgt_r_std + tgt_r_mean

    trans = cv2.merge([trans_b, trans_g, trans_r])  
    return trans


def fill_holes(im):
    if im.shape[-1] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    new_img = np.zeros_like(img).astype(np.uint8)
    img_template = np.expand_dims(new_img, axis=-1)
    img_template = np.tile(img_template, (1,1,3))
    img1 = np.zeros_like(img_template).astype(np.uint8)
    im = cv2.drawContours(img1, contours, -1, (255,255,255), cv2.FILLED)
    return im

def color_transition(image, mask, radius):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    result = image.copy()
    alpha = np.zeros_like(image, dtype=np.float32)
    alpha[mask == 255] = 1.0
    for y in range(radius, image.shape[0] - radius):
        for x in range(radius, image.shape[1] - radius):
            if mask[y, x] == 255:  
                surrounding_colors = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
                original_color = image[y, x]
                new_color = np.mean(surrounding_colors, axis=(0, 1))
                alpha_xy = np.linalg.norm(original_color - new_color) / np.linalg.norm(original_color)
                alpha[y, x] = alpha_xy
                result[y, x] = new_color.astype(np.uint8)

    return result, alpha

def subtract_fft_images(fg_dilated_mask, bg_dilated_mask, threshold=35, high_freq_thres=200):
    
    if fg_dilated_mask.shape[-1] == 3:
        fg_dilated_mask = cv2.cvtColor(fg_dilated_mask, cv2.COLOR_RGB2GRAY)
    if bg_dilated_mask.shape[-1] == 3:
        bg_dilated_mask = cv2.cvtColor(bg_dilated_mask, cv2.COLOR_RGB2GRAY)
    
    f1 = np.fft.fft2(fg_dilated_mask)
    f2 = np.fft.fft2(bg_dilated_mask)
    f1 = np.fft.fftshift(f1)
    f2 = np.fft.fftshift(f2)

    rows, cols = fg_dilated_mask.shape
    crow, ccol = rows // 2, cols // 2
    radius = high_freq_thres
    mask = np.zeros((rows, cols), np.uint8)
    center = (crow, ccol)
    cv2.circle(mask, center, radius, 1, -1)
    f1 = f1 * mask
    f2 = f2 * mask

    f = f1 - f2

    f = np.fft.ifftshift(f)
    img = np.fft.ifft2(f)
    img = np.abs(img)
    img = img.astype(np.uint8)
    img[img > threshold] = 255
    img[img <= threshold] = 0

    kernel = np.ones((21,21),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv_version = cv2.__version__.split('.')[0] # check cv2 version, to avoid unpack error

    if int(cv_version) >= 4:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    img1 = np.zeros_like(img).astype(np.uint8)
    img1 = np.expand_dims(img1, axis=-1)  
    img1 = np.tile(img1, (1, 1, 3))  
    img = cv2.drawContours(img1, contours, -1, (255,255,255), cv2.FILLED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


if __name__ == "__main__":
    fg = cv2.imread("/path/to/sty_edge.png", 0)
    bg = cv2.imread("/path/to/gt_edge.png", 0)
    subtract_fft_images()
