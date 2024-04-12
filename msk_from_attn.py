from typing import List
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
from math import ceil

from utils.mask_utils import *
from null_text_w_ptp import *
import ptp_utils



def aggregate_attention(attention_store: AttentionStore, prompts, res_h: int, res_w:int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention_period()
    num_pixels = res_h * res_w
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res_h, res_w, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_all_cross_attention(attention_store: AttentionStore, 
                             prompts: List[str], 
                             tokenizer = None,
                             from_where: List[str] = ['up','down'], 
                             original_resolution=(512, 512),
                             save_path='ca_vis'):
    tokens = tokenizer.encode(prompts[0])
    res_h, res_w = ceil(original_resolution[0]/32), ceil(original_resolution[1]/32)
    attention_maps = aggregate_attention(attention_store, [prompts[0]], res_h, res_w, from_where, True, 0)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((original_resolution[1], original_resolution[0])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), img_path=save_path+"/crossattn.png")

def show_diff_cross_attention(attention_store: AttentionStore, 
                             prompts: List[str], 
                             tokenizer= None,
                             from_where: List[str] = ['up','down'], 
                             original_resolution=(512, 512),
                             save_path='ca_vis'):
    tokens = tokenizer.encode(prompts[0])
    res_h, res_w = ceil(original_resolution[0]/32), ceil(original_resolution[1]/32)
    attention_maps = aggregate_attention(attention_store, [prompts[0]], res_h, res_w, from_where, True, 0)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        images.append(image)
    results = []
    new_image = cv2.subtract(images[0], images[-1])
    new_image = np.array(Image.fromarray(new_image).resize((original_resolution[1], original_resolution[0])))
    results.append(new_image)

    ptp_utils.view_images(np.stack(results, axis=0), img_path=save_path+"/crossattn_diff.png")

    new_image = cv2.convertScaleAbs(new_image, alpha=2, beta=0)
    thres = 108
    new_image[new_image<=thres] = 0
    new_image[new_image>thres] = 255
    
    new_image = cv2.bitwise_not(new_image)
    return new_image

def show_self_attention_comp(attention_store: AttentionStore, 
                             prompts,
                             from_where: List[str],
                             max_com=5, 
                             original_resolution=(512, 512),
                             save_path='ca_vis'
    ):
    res_h, res_w = ceil(original_resolution[0]/32), ceil(original_resolution[1]/32)
    attention_maps = aggregate_attention(attention_store, [prompts[0]], res_h, res_w, from_where, False, 0).numpy().reshape((res_h ** 2, res_w ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res_h, res_w)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((original_resolution[1], original_resolution[0]))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), img_path=save_path+'/selfattn_diff.png')


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def tensor2label(tensor):
    label_list = []
    mapper = {'0':'CHANGE', '1':'ADD', '2':'REMOVE'}
    if len(tensor.shape) == 2:
        for i in range(len(tensor)):
            label_list.append(mapper[str(tensor[i].argmax().item())])
    elif len(tensor.shape) == 1:
        label_list.append(mapper[str(tensor.argmax().item())])
    return label_list

def cls_predict(model, tokenizer, text_encoder, text=""):
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to('cuda')
        input_ids = encodings['input_ids']
        attn_mask = encodings['attention_mask']
        embeddings = text_encoder(input_ids, attn_mask)[0]
        outputs = model(embeddings)
        _, predicted = torch.max(outputs, 1)
        predicted_labels = tensor2label(predicted)
    return predicted_labels[0]

def action_classify(ip2p_pipe, prompt):
    cls_model = MLPClassifier(input_size=ip2p_pipe.text_encoder.config.hidden_size, hidden_size=128, num_classes=3).to('cuda')
    cls_model.load_state_dict(
        torch.load("./ckpts/action_classifier.pth")
    )
    mode = cls_predict(cls_model, ip2p_pipe.tokenizer, ip2p_pipe.text_encoder, prompt)
    del cls_model
    torch.cuda.empty_cache()
    return mode



