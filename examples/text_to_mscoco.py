import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image

from latent_sdxl import get_solver as get_solver_sdxl
from utils.log_util import create_workdir, set_seed

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="JEPA-guided SDXL Sampling")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/mscoco")
    parser.add_argument('--prompt_dir', type=Path, default=Path('examples/assets/coco_v2.txt'))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default='ddim_jepa',
                        choices=["ddim_jepa", "ddim_jepa_lightning"])
    parser.add_argument("--model", type=str, default='sdxl', choices=["sdxl", "sdxl_lightning"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--ddim_eta", type=float, default=0.0)

    # JEPA parameters
    parser.add_argument("--use_jepa", action='store_true')
    parser.add_argument("--jepa_backbone", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "metaclip"],
                        help="Backbone for JEPA guidance")
    parser.add_argument("--jepa_eta", type=float, default=0.5)
    parser.add_argument("--g_interval", type=int, default=1)
    parser.add_argument("--g_start_t", type=float, default=0.8)
    parser.add_argument("--rsvd_topk", type=int, default=9)
    parser.add_argument("--rsvd_pi_q", type=int, default=2)
    parser.add_argument("--rsvd_oversample", type=int, default=2)
    parser.add_argument("--use_normed_grad", action='store_true')
    parser.add_argument("--jg_img_size", type=int, default=224)
    parser.add_argument("--jg_schedule", type=str, default="constant", choices=["constant", "variance"])

    args = parser.parse_args()

    jepa_config = {
        'use_jepa': args.use_jepa,
        'jepa_backbone': args.jepa_backbone,
        'jepa_eta': args.jepa_eta,
        'g_interval': args.g_interval,
        'g_start_t': args.g_start_t,
        'rsvd_topk': args.rsvd_topk,
        'rsvd_pi_q': args.rsvd_pi_q,
        'rsvd_oversample': args.rsvd_oversample,
        'use_normed_grad': args.use_normed_grad,
        'jg_img_size': args.jg_img_size,
        'jg_schedule': args.jg_schedule,
        'seed': args.seed,
    }

    set_seed(args.seed)
    create_workdir(args.workdir)

    # load prompts
    text_list = []
    with open(args.prompt_dir, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                text_list.append(stripped)
    text_list = text_list[args.resume_from : (args.resume_from + args.num_samples)]

    solver_config = munchify({'num_sampling': args.NFE})

    if args.model == "sdxl":
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device,
                                 seed=args.seed)
    else:  # sdxl_lightning
        light_model_ckpt = f"ckpt/sdxl_lightning_{args.NFE}step_unet.safetensors"
        print(f"Using light model checkpoint: {light_model_ckpt}")
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device,
                                 light_model_ckpt=light_model_ckpt,
                                 seed=args.seed)

    solver.setup_jepa(jepa_config)

    img_count = args.resume_from
    for i, text in enumerate(tqdm(text_list, desc='Sampling')):
        print(f'Processing {i+1}/{len(text_list)}: {text}')
        result = solver.sample(
            prompt1=[args.null_prompt, text],
            prompt2=[args.null_prompt, text],
            cfg_guidance=args.cfg_guidance,
            target_size=(1024, 1024),
            ddim_eta=args.ddim_eta,
        )
        save_image(result, args.workdir.joinpath(f'{str(img_count).zfill(5)}.png'), normalize=True)
        img_count += 1


if __name__ == "__main__":
    main()
