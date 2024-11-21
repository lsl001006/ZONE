from msk_from_attn import *
from ptp_utils import check_image_size
from diffusers import DDIMScheduler,StableDiffusionInstructPix2PixPipeline,DPMSolverMultistepScheduler
from null_text_w_ptp import text2image_instructpix2pix_blend
import argparse
from PIL import ImageEnhance

def model_init_insert(args):
    t1 = time()

    ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.instructpix2pix_path, 
            safety_checker=None)
    ip2p_pipe.to("cuda")
    ip2p_pipe.scheduler = DPMSolverMultistepScheduler.from_config(ip2p_pipe.scheduler.config)

    sam_predictor = sam_model_registry["vit_h"](
        checkpoint=args.sam_ckpt_path,
    ).to('cuda')

    sd_pipe = None

    mb_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.magicbrush_path,
        safety_checker=None)
    mb_pipe.to("cuda")
    mb_pipe.scheduler = DPMSolverMultistepScheduler.from_config(ip2p_pipe.scheduler.config)
   
    t2 = time()
    print(f'>>> Initializing used {t2-t1:.2f} s.')

    return sam_predictor, sd_pipe, ip2p_pipe, mb_pipe

    
def fused_ip2p(args):
    # 1. initialize pretrained models
    sam_predictor, _, pipe, mb_pipe = model_init_insert(args)
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path+'/composite', exist_ok=True)
    os.makedirs(output_path+'/bimask', exist_ok=True)
    os.makedirs(output_path+'/ca_vis_bld', exist_ok=True)
    os.makedirs(output_path+'/sty', exist_ok=True)
    os.makedirs(output_path+'/rgba', exist_ok=True)
    
    prompts = [args.instruction]
    image_name = os.path.basename(args.image_path).split('.')[0]
    gt_image_pil = Image.open(args.image_path)
    # update: check image size && resize to avoid OOM
    gt_image_pil = check_image_size(gt_image_pil)
    w, h = gt_image_pil.size
    mode = action_classify(pipe, args.instruction)
    tokenizer = pipe.tokenizer
    print('[RUN MODE]: ',mode)

    controller_bld = AttentionStore(start_step=0)
    generator = torch.Generator().manual_seed(args.seed)
    style_image, _ = text2image_instructpix2pix_blend(
                                                    pipe, 
                                                    mb_pipe,
                                                    mode,
                                                    gt_image_pil, 
                                                    prompts, 
                                                    controller_bld, 
                                                    guidance_scale=args.image_guidance_scale, 
                                                    generator=generator, 
                                                    num_inference_steps=args.inference_steps,
                                                    beta=args.blend_beta)

    show_all_cross_attention(controller_bld, prompts, tokenizer, original_resolution=(h,w), save_path=f'{output_path}/ca_vis_bld')
    mask = show_diff_cross_attention(controller_bld, prompts, tokenizer, original_resolution=(h,w), save_path=f'{output_path}/ca_vis_bld')
    cv2.imwrite(f'{output_path}/bimask/{image_name}_bld.png', mask)
    style_image.save(f'{output_path}/sty/{image_name}_bld.png')
    
    enhancer = ImageEnhance.Brightness(style_image)
    style_image = enhancer.enhance(args.brightness)

    t1 = time()
    if not mode == "REMOVE":
        masked_out, mask = get_finemask_everything(mask, style_image, sam_predictor)
    else:
        masked_out, mask = get_finemask_everything(mask, gt_image_pil, sam_predictor)
    masked_out = cv2.cvtColor(masked_out, cv2.COLOR_BGR2RGB)
    t2 = time()
    print(f'Fine mask prediction used {t2-t1:.2f} s')
    cv2.imwrite(f'{output_path}/bimask/{image_name}_fine.png', mask)
    
    # 6. dilate mask for fine composition
    mask_image = mask
    dilated_mask_all, _ = dilate_mask(mask_image, iterations=args.dilate_strength)
    cv2.imwrite(f'{output_path}/bimask/{image_name}_dilate.png', dilated_mask_all)
    
    # 7. ops based on mode
    style_image_np = cv2.cvtColor(np.array(style_image), cv2.COLOR_BGR2RGB)
    gt_image_np = cv2.cvtColor(np.array(gt_image_pil), cv2.COLOR_BGR2RGB)
    canvas = gt_image_np.copy()
    
    if mode == 'ADD':
        print('>>Running ADD mode')
        canvas[mask_image>0] = masked_out[mask_image>0]
    elif mode == 'REMOVE':
        print('>>Running REMOVE mode')
        masked_out = cv2.cvtColor(np.array(style_image), cv2.COLOR_BGR2RGB)
        masked_out[dilated_mask_all==0]=0
        canvas[dilated_mask_all>0] = masked_out[dilated_mask_all>0]
    elif mode == 'CHANGE':
        new_masked_out = cv2.cvtColor(np.array(style_image), cv2.COLOR_BGR2RGB)
        new_masked_out[dilated_mask_all==0]=0
        canvas[mask_image>0]= new_masked_out[mask_image>0]
        

    cv2.imwrite(f'{output_path}/composite/{image_name}.png', canvas)
    cv2.imwrite(f'{output_path}/rgba/{image_name}.png', masked_out)
    

    gt_edge = gt_image_np.copy()
    sty_edge = style_image_np.copy()
    gt_edge[dilated_mask_all==0] = 0
    sty_edge[dilated_mask_all==0] = 0

    # edge smoother
    modified_sty_edge = subtract_fft_images(sty_edge, gt_edge, threshold=args.threshold)
    new_sty_edge = style_image_np.copy()
    new_sty_edge[modified_sty_edge==0] = 0
    edge_bimask = modified_sty_edge - mask_image
    
    new_canvas = gt_image_np.copy()
    new_canvas[modified_sty_edge>0] = style_image_np[modified_sty_edge>0]
    new_canvas[edge_bimask>0] = style_image_np[edge_bimask>0]*args.alpha + gt_image_np[edge_bimask>0]*(1-args.alpha)
    cv2.imwrite(f'{output_path}/composite/{image_name}_fft.png', new_canvas)

    print('-'*20, ' Inference completed ', '-'*20)


def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--instruction', type=str, default="Make the dog a golden statue", help='The instruction')
    parser.add_argument('--image_path', type=str, default="./data/21.png", help='path to image')
    parser.add_argument('--sam_ckpt_path', type=str, default="./ckpts/sam_vit_h_4b8939.pth", help='The path to SAM checkpoint')
    parser.add_argument('--instructpix2pix_path', type=str, default="timbrooks/instruct-pix2pix", help='The path to InstructPix2Pix checkpoint')
    parser.add_argument('--magicbrush_path', type=str, default="vinesmsuic/magicbrush-jul7", help='The path to MagicBrush checkpoint')
    parser.add_argument('--output_path', type=str, default='./outputs', help='The output path')
    parser.add_argument('--threshold', type=int, default=30, help='The threshold of edge smoother')
    parser.add_argument('--alpha', type=float, default=0.6, help='The alpha value for blending style image and original image')
    parser.add_argument('--blend_beta', type=float, default=0.2, help='The beta value for fusing IP2P and MB')
    parser.add_argument('--image_guidance_scale', type=float, default=7.5, help='The image guidance scale')
    parser.add_argument('--inference_steps', type=int, default=20, help='The inference steps')
    parser.add_argument('--dilate_strength', type=int, default=4, help='The dilate strength')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--brightness', type=float, default=1, help='The brightness of the style image')
    
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = parse_args()
    fused_ip2p(args)
