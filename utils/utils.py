import torch
from null_text_w_ptp import *
from torch.optim.adam import Adam
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import os

@torch.no_grad()
def inpaint_sd(
    pipe,
    prompt = 'high quality image',
    image = None,
    mask_image = None,
    masked_image_latents: torch.FloatTensor = None,
    height = None,
    width = None,
    strength: float = 0.1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt = None,
    eta: float = 0.0,
    generator = None,
    latents = None,
    output_type = "pil",
):
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
    pipe.check_inputs(
        prompt,
        height,
        width,
        strength,
        1,
        negative_prompt,
        None,
        None,
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    device = pipe._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        1,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
    )

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps=num_inference_steps, strength=strength, device=device
    )
    
    if num_inference_steps < 1:
        raise ValueError(
            f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
            f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
        )
    
    latent_timestep = timesteps[:1].repeat(batch_size * 1)
    is_strength_max = strength == 1.0

    mask, masked_image, init_image = prepare_mask_and_masked_image(
        image, mask_image, height, width, return_image=True
    )

    num_channels_latents = pipe.vae.config.latent_channels
    num_channels_unet = pipe.unet.config.in_channels
    return_image_latents = num_channels_unet == 4
    latents_outputs = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=return_image_latents,
    )

    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs

    mask, masked_image_latents = pipe.prepare_mask_latents(
        mask,
        masked_image,
        batch_size,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        do_classifier_free_guidance,
    )
    init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
    init_image = pipe._encode_vae_image(init_image, generator=generator)

    if num_channels_unet == 9:
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif num_channels_unet != 4:
        raise ValueError(
            f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
        )

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        if num_channels_unet == 9:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        if num_channels_unet == 4:
            init_latents_proper = image_latents[:1]
            init_mask = mask[:1]

            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = pipe.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - init_mask) * init_latents_proper + init_mask * latents

    if not output_type == "latent":
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    else:
        image = latents

    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    latents = pipe.image_processor.postprocess(latents, output_type='latent', do_denormalize=do_denormalize)
    
    return image[0], latents


def show_merged_crossattn_mask(attention_store: AttentionStore, 
                               from_where: List[str], 
                               words: List[str], 
                               prompts: List[str], 
                               original_resolution=(512, 512), 
                               thres = 128,
                               save_path='ca_vis',
                               resize_factor=1):
    tokens = tokenizer.encode(prompts[0])
    decoder = tokenizer.decode
    res_h, res_w = ceil(original_resolution[0]/32), ceil(original_resolution[1]/32)
    attention_maps = aggregate_attention(attention_store, [prompts[0]], res_h, res_w, from_where, True, 0)
    inds = []
    for word in words:
        ind = ptp_utils.get_word_inds(prompts[0], word, tokenizer)
        if ind.size == 0:
            print(f'[ERROR] Word: {word} do not exist in {prompts[0]}, check if `,` or `.` besides the word!')
            continue
        if ind.size > 1:
            ind = np.array([ind[1]])
        inds.append(ind)
    if len(inds) == 0:
        print('[ERROR] No valid word detected in the prompts, please check.')
        return
    inds = np.array(inds)
    images = attention_maps[:, :, inds]
    if len(images.shape) > 3:
        images = images.squeeze(-1)

    for j in range(images.shape[-1]):
        images[:,:,j] = 255 * images[:,:,j] / images[:,:,j].max()
    new_images = images.sum(-1).unsqueeze(-1)
    new_images = new_images/new_images.max()*255)
    new_images = new_images.repeat(1,1,3)
    mask = new_images.numpy().astype(np.uint8)
    mask = np.array(Image.fromarray(mask).resize((original_resolution[1]*resize_factor, original_resolution[0]*resize_factor)))

    ptp_utils.view_images(mask, img_path=save_path+'/'+words[0]+'.png')
    print(f'cross attention mask for {words} is saved to {save_path}/{words[0]}.png')

    mask[mask > thres] = 255
    mask[mask <= thres] = 0
    return mask
    
def show_exclued_crossattn_mask(attention_store: AttentionStore, 
                                from_where: List[str], 
                                words: List[str], 
                                exclued_objects: List[str], 
                                prompts: List[str], 
                                original_resolution=(512, 512), 
                                thres = 128,
                                save_path="ca_vis",
                                resize_factor=1):
    tokens = tokenizer.encode(prompts[0])
    decoder = tokenizer.decode
    res_h, res_w = ceil(original_resolution[0]/32), ceil(original_resolution[1]/32)
    attention_maps = aggregate_attention(attention_store, [prompts[0]], res_h, res_w, from_where, True, 0) 
    inds = []
    exclued_inds = []
    for word in words:
        ind = ptp_utils.get_word_inds(prompts[0], word, tokenizer)
        if ind.size == 0:
            print(f'[ERROR] Word: {word} do not exist in {prompts[0]}, check if `,` or `.` besides the word!')
            continue
        inds.append(ind)
    if len(inds) == 0:
        print('[ERROR] No valid word detected in the prompts, please check.')
        return

    for word in exclued_objects:
        ind = ptp_utils.get_word_inds(prompts[0], word, tokenizer)
        if ind.size == 0:
            print(f'[ERROR] Word: {word} do not exist in {prompts[0]}, check if `,` or `.` besides the word!')
            continue
        exclued_inds.append(ind)
    

    inds = np.array(inds)
    images = attention_maps[:, :, inds] 
    if len(images.shape) > 3:
        images = images.squeeze(-1)
    for j in range(images.shape[-1]):
        images[:,:,j] = 255 * images[:,:,j] / images[:,:,j].max()

    focus_images = images.sum(-1).unsqueeze(-1)
    focus_images = focus_images/focus_images.max()*255
    new_images = focus_images.clone()

    if len(exclued_inds) > 0:
        exclued_inds = np.array(exclued_inds)
        images = attention_maps[:, :, exclued_inds]
        if len(images.shape) > 3:
            images = images.squeeze(-1)
        for j in range(images.shape[-1]):
            images[:,:,j] = 255 * images[:,:,j] / images[:,:,j].max()
        dismiss_images = images.sum(-1).unsqueeze(-1)
        dismiss_images = dismiss_images/dismiss_images.max()*255

        new_images = focus_images - dismiss_images
        new_images = torch.clamp(new_images, min=0) 
        new_images = new_images/new_images.max()*255 
        thres=thres/4
    
    new_images = new_images.repeat(1,1,3)
    mask = new_images.numpy().astype(np.uint8)
    mask = np.array(Image.fromarray(mask).resize((original_resolution[1]*resize_factor, original_resolution[0]*resize_factor)))
    ptp_utils.view_images(mask, img_path=save_path+'/'+str(words)+'_'+str(exclued_objects)+'.png')
    print(f'excluded cross attention mask for {word} is saved to {save_path}/{words}_{exclued_objects}.png')

    mask[mask > thres] = 255
    mask[mask <= thres] = 0
    return mask

def NTIProcess(model, gt_image, image_name, prompts, output_path, resize_factor=1):
    if isinstance(gt_image, Image.Image):
        gt_image = np.array(gt_image)
    null_inversion = NullInversion(model)
    load_nti_npy = f'{output_path}/nti_pkls/{image_name}.pkl'
    original_gt_image = gt_image.copy()
    
    h, w = gt_image.shape[0]//resize_factor, gt_image.shape[1]//resize_factor
    gt_image = cv2.resize(gt_image, (w, h)) 

    if os.path.exists(load_nti_npy):
        with open(load_nti_npy, 'rb') as f:
            data = pickle.load(f)
        x_t = data['x_t']
        uncond_embeddings = data['uncond']
        prompt = data['prompt']
        if prompt != prompts[0]:
            (_, _), x_t, uncond_embeddings = null_inversion.invert(gt_image, prompts[0], offsets=(0,0,0,0), num_inner_steps=5, verbose=True)
            data = {'x_t':x_t, 'uncond':uncond_embeddings, 'prompt':prompts[0]}
            with open(load_nti_npy, 'wb') as f:
                pickle.dump(data, f)
    else:
        (_, _), x_t, uncond_embeddings = null_inversion.invert(gt_image, prompts[0], offsets=(0,0,0,0), num_inner_steps=5, verbose=True)
        data = {'x_t':x_t, 'uncond':uncond_embeddings, 'prompt':prompts[0]}
        with open(load_nti_npy, 'wb') as f:
            pickle.dump(data, f)
    return x_t, uncond_embeddings