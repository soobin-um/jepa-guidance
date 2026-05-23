from typing import Any, Optional, Tuple
import os
from safetensors.torch import load_file

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)
from tqdm import tqdm
import copy

####### Factory #######
__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

########################

class SDXL():
    def __init__(self,
                 solver_config: dict,
                 model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 dtype=torch.float16,
                 device='cuda',
                 seed: int=42):

        self.device = device
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1_base = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2_base = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1_base = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2_base = copy.deepcopy(pipe.text_encoder_2)
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

        # a dedicated generator for various purposes
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None):
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1

        prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            # Comment on diffusers' source code:
            # "We are only ALWAYS interested in the pooled output of the final text encoder"
            # i.e. we overwrite the pool_prompt_embed with the new one
            prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]

        null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, zt):
        image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
        return image

    def predict_noise(self, zt, t, uc, c, added_cond_kwargs):
        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
        add_time_ids = list(original_size+crops_coords_top_left+target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        assert expected_add_embed_dim == passed_add_embed_dim, (
             f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", ""],
               prompt2 = ["", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               **kwargs):

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # reset tokenizer and text_encoder
        self.tokenizer_1 = copy.deepcopy(self.tokenizer_1_base)
        self.tokenizer_2 = copy.deepcopy(self.tokenizer_2_base)
        self.text_enc_1 = copy.deepcopy(self.text_enc_1_base)
        self.text_enc_2 = copy.deepcopy(self.text_enc_2_base)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         pool_null_embed,
         pool_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds, prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          add_cond_kwargs: Optional[dict]=None,
                          **kwargs):
        if method == 'ddim':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               kwargs.get('cfg_guidance', 0.0),
                               add_cond_kwargs)
        elif method == 'npi':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               1.0,
                               add_cond_kwargs)
        elif method == 'random':
            size = kwargs.get('size', (1, 4, 128, 128))
            z = torch.randn(size).to(self.device)
        else:
            raise NotImplementedError

        return z.requires_grad_()

    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted.
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    def reverse_process(self, *args, **kwargs):
        raise NotImplementedError


class SDXLLightning(SDXL):
    def __init__(self,
                 solver_config: dict,
                 base_model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 light_model_ckpt:str="ckpt/sdxl_lightning_4step_unet.safetensors",
                 dtype=torch.float16,
                 device='cuda',
                 seed: int = 42):

        self.device = device

        # load the student model
        unet = UNet2DConditionModel.from_config(base_model_key, subfolder="unet").to("cuda", torch.float16)
        ext = os.path.splitext(light_model_ckpt)[1]
        if ext == ".safetensors":
            state_dict = load_file(light_model_ckpt)
        else:
            state_dict = torch.load(light_model_ckpt, map_location="cpu")
        print(unet.load_state_dict(state_dict, strict=True))
        unet.requires_grad_(False)
        self.unet = unet

        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_key, unet=self.unet, torch_dtype=dtype).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1_base = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2_base = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1_base = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2_base = copy.deepcopy(pipe.text_encoder_2)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)

        # a dedicated generator for various purposes
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)


###########################################
# JEPA guidance
###########################################

@register_solver("ddim_jepa")
class DDIMWithJEPA(SDXL):
    """
    DDIM solver with JEPA guidance for SDXL
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_jepa = None
        self.jepa_rng = None
        self.jepa_config = {}

    def setup_jepa(self, jepa_config):
        """Initialize JEPA model and config"""
        # Save global RNG state (torch.hub.load changes it)
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        try:
            self.jepa_config = jepa_config
            self.jg_img_size = self.jepa_config.get('jg_img_size', 224)
            self.seed = self.jepa_config.get('seed', 42)
            jepa_backbone = self.jepa_config.get('jepa_backbone', 'dinov2_vits14')

            if jepa_config.get('use_jepa', False):
                if 'dinov2' in jepa_backbone.lower():
                    # Disable efficient/flash SDPA for DINOv2
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)

                    self.f_jepa = torch.hub.load(
                        'facebookresearch/dinov2', f'{jepa_backbone.lower()}_reg'
                    ).to(self.device).eval()
                    print(f"[JEPA] Initialized {jepa_backbone} for JEPA guidance")
                elif 'metaclip' in jepa_backbone.lower():
                    # Disable memory-efficient attention backends for MetaCLIP gradient computation
                    print("[JEPA] Disabling flash/mem_efficient attention backends for MetaCLIP gradient computation...")
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)

                    import open_clip
                    backbone, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-16-quickgelu', pretrained='metaclip_400m'
                    )
                    self.f_jepa = backbone.visual.to(self.device).eval()
                    print("[JEPA] Initialized MetaCLIP for JEPA guidance")
                else:
                    raise ValueError(f"Unknown jepa_backbone: {jepa_backbone}. Choose 'dinov2_*' or 'metaclip'")

                self.jepa_rng = torch.Generator(device=self.device)
                self.jepa_rng.manual_seed(self.seed)
        finally:
            # Restore global RNG state
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    def compute_jepa_gradient(self, zt, step, total_steps, t=None, at=None, at_prev=None,
                              uc=None, c=None, add_cond_kwargs=None, cfg_guidance=7.5):
        """Compute JEPA score gradient w.r.t. noisy latent zt"""
        cfg = self.jepa_config
        eta = cfg.get('jepa_eta', 1.0)
        g_interval = cfg.get('g_interval', 3)
        g_start_t = cfg.get('g_start_t', 0.8)
        k = cfg.get('rsvd_topk', 3)
        q_steps = cfg.get('rsvd_pi_q', 2)
        p = cfg.get('rsvd_oversample', 2)
        use_normed_grad = cfg.get('use_normed_grad', True)
        jg_img_size = cfg.get('jg_img_size', 224)
        jg_schedule = cfg.get('jg_schedule', 'variance')
        jepa_backbone = cfg.get('jepa_backbone', 'dinov2_vits14')
        eps = 1e-8

        # Normalization stats based on backbone type
        if 'dinov2' in jepa_backbone.lower():
            # ImageNet normalization stats for DINOv2
            norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(zt.device)
            norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(zt.device)
        elif 'metaclip' in jepa_backbone.lower():
            # CLIP normalization stats for MetaCLIP
            norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(zt.device)
            norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(zt.device)
        else:
            raise ValueError(f"Unknown jepa_backbone: {jepa_backbone}. Choose 'dinov2_*' or 'metaclip'")

        # Check timing (t_ratio: 1.0 at start -> 0.0 at end)
        t_ratio = 1.0 - (step / total_steps)
        if step % g_interval != 0 or t_ratio > g_start_t:
            return None

        r = k + p

        with torch.enable_grad():
            zt_in = zt.detach().clone().requires_grad_(True)

            # Re-compute noise_pred with zt_in to build computational graph
            noise_uc, noise_c = self.predict_noise(zt_in, t, uc, c, add_cond_kwargs)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Predict clean latent from noisy latent
            z0t = (zt_in - (1-at).sqrt() * noise_pred) / at.sqrt()

            # Decode latent -> image [0,1]
            vae_dtype = next(self.vae.parameters()).dtype
            z_scaled = (z0t / self.vae.config.scaling_factor).to(vae_dtype)
            img = self.vae.decode(z_scaled).sample
            x0p = (img / 2 + 0.5).clamp(0, 1).float()

            # Resize and normalize for JEPA backbone
            x0p = F.interpolate(x0p, size=(jg_img_size, jg_img_size), mode="bilinear", align_corners=False)
            x0p = (x0p - norm_mean) / norm_std

            B, C, Hp, Wp = x0p.shape

            # Jx224
            f_base = self.f_jepa(x0p)

            # Manual Jv/JTu to avoid xFormers/SDPA jvp incompatibility
            def Jv(v, create_graph=False):
                # Forward-mode AD via finite difference: (f(x + eps*v) - f(x)) / eps
                eps_fd = 1e-4
                with torch.no_grad():
                    f_plus = self.f_jepa(x0p + eps_fd * v)
                return (f_plus - f_base.detach()) / eps_fd

            def JTu(u, create_graph=False):
                # Reverse-mode AD via standard backward
                # Use x0p directly to maintain graph connection to zt_in
                grad = torch.autograd.grad(f_base, x0p, grad_outputs=u,
                                          create_graph=create_graph, retain_graph=True)[0]
                return grad

            # Retry loop: resample Omega if SVD fails
            max_svd_retries = 3
            for svd_attempt in range(max_svd_retries):
                # 1) Random Omega
                Omega = torch.randn(B, r, C, Hp, Wp, device=x0p.device, dtype=x0p.dtype,
                                   generator=self.jepa_rng)
                Omega = Omega / (Omega.view(B, r, -1).norm(dim=2, keepdim=True).view(B, r, 1, 1, 1) + eps)

                # 2) Y = J @ Omega
                Y_cols = [Jv(Omega[:, j], create_graph=False) for j in range(r)]
                Y = torch.stack(Y_cols, dim=2)

                # 2.5) Subspace iteration
                for _ in range(q_steps):
                    Y_cols = []
                    for j in range(r):
                        wj = JTu(Y[:, :, j].detach(), create_graph=False)
                        Y_cols.append(Jv(wj, create_graph=False))
                    Y = torch.stack(Y_cols, dim=2)
                    # QR decomposition requires float32
                    Y_float = Y.float()
                    Y_float, _ = torch.linalg.qr(Y_float, mode="reduced")
                    Y = Y_float.to(Y.dtype)

                # QR decomposition requires float32
                Y_float = Y.float()
                Q_float, _ = torch.linalg.qr(Y_float, mode="reduced")
                Q = Q_float.to(Y.dtype).detach()

                # This avoids Jv (which has no grad graph with finite diff)
                JTQ_cols = []
                for j in range(r):
                    qj = Q[:, :, j]
                    wj = JTu(qj, create_graph=True)  # J^T @ q_j, shape (B, C, H, W)
                    JTQ_cols.append(wj)
                # JTQ: list of r tensors, each (B, C, H, W)
                # Stack into (B, r, C, H, W) then flatten spatial dims
                JTQ = torch.stack(JTQ_cols, dim=1)  # (B, r, C, H, W)
                JTQ_flat = JTQ.view(B, r, -1)  # (B, r, C*H*W)

                # 4) Singular values -> JEPA loss (requires float32)
                # SVD is more numerically stable than eigendecomposition of M = JTQ @ JTQ^T
                JTQ_float = JTQ_flat.float()
                try:
                    sigmas = torch.linalg.svdvals(JTQ_float)  # (B, r), descending order
                    break  # success
                except torch._C._LinAlgError:
                    try:
                        # CUDA SVD failed, fallback to CPU (more robust algorithm)
                        sigmas = torch.linalg.svdvals(JTQ_float.cpu()).to(JTQ_float.device)
                        break  # success
                    except torch._C._LinAlgError:
                        if svd_attempt < max_svd_retries - 1:
                            print(f"[JEPA] SVD failed, resampling Omega (attempt {svd_attempt + 1}/{max_svd_retries})")
                            continue
                        else:
                            print(f"[JEPA] SVD failed after {max_svd_retries} attempts, skipping this step")
                            return None

            sigmas_top = torch.clamp(sigmas[:, :k], min=eps)
            # Sum over k (singular values) but keep batch dimension
            jepa_loss_per_sample = torch.log(sigmas_top).sum(dim=1)  # (B,)
            jepa_loss = jepa_loss_per_sample.sum()  # scalar for backward

            grad = torch.autograd.grad(jepa_loss, zt_in)[0]

        if use_normed_grad:
            max_grad = grad.abs().amax(dim=(1, 2, 3), keepdim=True)
            grad = grad / (max_grad + eps)

        # Compute variance scheduling
        if jg_schedule == 'variance':
            # Variance of the reverse process: variance = (1 - at_prev) / (1 - at) * (1 - at/at_prev)
            variance = ((1 - at_prev) / (1 - at) * (1 - at / at_prev)).clamp(min=eps)
            print(f"JEPA scaling at step {step}:", variance.item())
            final_grad = eta * variance * grad
            return final_grad
        elif jg_schedule == 'constant':
            print("JEPA constant scaling at step", step)
            return eta * grad

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        ddim_eta: float = 0.0,
                        **kwargs):

        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        total_steps = len(self.scheduler.timesteps)
        use_jepa = self.jepa_config.get('use_jepa', False) and self.f_jepa is not None

        if cfg_guidance == 1.0:
            null_prompt_embeds = None

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL+JEPA' if use_jepa else 'SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            # JEPA guidance on zt
            if use_jepa:
                jepa_grad = self.compute_jepa_gradient(
                    zt, step, total_steps, t=t, at=at, at_prev=at_next,
                    uc=null_prompt_embeds, c=prompt_embeds,
                    add_cond_kwargs=add_cond_kwargs, cfg_guidance=cfg_guidance
                )
                if jepa_grad is not None:
                    zt = zt - jepa_grad
                    pbar.set_postfix({'jepa': 'applied'})

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            if ddim_eta > 0.0:
                sigma_t = ddim_eta * torch.sqrt((1 - at_next) / (1 - at) * (1 - at / at_next))
                noise_rand = torch.randn_like(zt) * sigma_t
                zt = at_next.sqrt() * z0t + (1-at_next-sigma_t**2).sqrt() * noise_pred + noise_rand
            else:
                zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        return z0t


@register_solver("ddim_jepa_lightning")
class DDIMWithJEPALight(DDIMWithJEPA, SDXLLightning):
    """
    DDIM solver with JEPA guidance for SDXL Lightning
    """
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
        self.f_jepa = None
        self.jepa_rng = None
        self.jepa_config = {}

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        ddim_eta: float = 0.0,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return DDIMWithJEPA.reverse_process(
            self,
            null_prompt_embeds,
            prompt_embeds,
            cfg_guidance,
            add_cond_kwargs,
            shape,
            callback_fn,
            ddim_eta=ddim_eta,
            **kwargs
        )


if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
