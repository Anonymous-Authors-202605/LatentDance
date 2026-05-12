#!/bin/bash

### single gpu test command
torchrun --nnodes=1 --nproc_per_node=1 scripts/v8_general/test_batch.py \
    --sft_model_high "checkpoints/LatentDance/latentdance_high_noise.safetensors" \
    --sft_model_low "checkpoints/LatentDance/latentdance_low_noise.safetensors" \
    --local_model_path "./checkpoints" \
    --reference_image "data/evaldata/input_image" \
    --control_video "data/evaldata/pose3" \
    --prompt "best quality, 8k, masterpiece, highly detailed" \
    --node_rank 0 \
    --num_nodes 1 \
    --first_as_guidance \
    --fix_missing_warp_v2 \
    --score_filter \
    --warp_limbs \
    --face_skip \
    --vis_warp_keypoints \
    --key_points "data/evaldata/pose3_keypoints" \
    --caption_csv "data/evaldata/pllava_caption/caption.csv" \
    --node_rank 0 --num_nodes 1

### multi gpu test command
# torchrun --nnodes=1 --nproc_per_node=8 scripts/v8_general/test_batch.py --use_usp\
#     --sft_model_high "checkpoints/LatentDance/latentdance_high_noise.safetensors" \
#     --sft_model_low "checkpoints/LatentDance/latentdance_low_noise.safetensors" \
#     --local_model_path "./checkpoints" \
#     --reference_image "data/evaldata/input_image" \
#     --control_video "data/evaldata/pose3" \
#     --prompt "best quality, 8k, masterpiece, highly detailed" \
#     --node_rank 0 \
#     --num_nodes 1 \
#     --first_as_guidance \
#     --fix_missing_warp_v2 \
#     --score_filter \
#     --warp_limbs \
#     --face_skip \
#     --vis_warp_keypoints \
#     --key_points "data/evaldata/pose3_keypoints" \
#     --caption_csv "data/evaldata/pllava_caption/caption.csv" \
#     --node_rank 0 --num_nodes 1


