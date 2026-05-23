gpu=0

model="sdxl_lightning"
method="ddim_jepa_lightning"
cfg_w=1.0
NFE=4

seed=42
num_samples=5000

##### JEPA parameters used for Table 2 #####
jepa_eta=0.5
g_interval=1
g_start_t=0.8
rsvd_topk=9
rsvd_pi_q=2
rsvd_oversample=2
jg_img_size=224
jg_schedule="constant"

savedir="results/sdxl-lt_jepa"

conda activate jepa-guidance

CUDA_VISIBLE_DEVICES=$gpu python -m examples.text_to_mscoco \
    --workdir "$savedir" \
    --method "$method" --NFE $NFE --cfg_guidance $cfg_w --model "$model" \
    --seed $seed --num_samples $num_samples \
    --use_jepa --jepa_eta $jepa_eta --g_interval $g_interval --g_start_t $g_start_t \
    --rsvd_topk $rsvd_topk --rsvd_pi_q $rsvd_pi_q --rsvd_oversample $rsvd_oversample --use_normed_grad --jg_img_size $jg_img_size \
    --jg_schedule $jg_schedule

# JEPA scoring for generated images
CUDA_VISIBLE_DEVICES=$gpu python jepa_scoring.py \
    --img_root "$savedir" --device cuda
    
conda deactivate
    