import torch
import numpy as np
import csv
from diffsynth.utils.data import save_video,VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.loader.file import load_state_dict
from PIL import Image
from modelscope import dataset_snapshot_download
import argparse
import os

# Import LoadKeypoints, LoadDepthKeypoints, and LoadDepthKeypoints2 from train.py
from train import LoadKeypoints, LoadDepthKeypoints, LoadDepthKeypoints2

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _detect_num_kps_from_path(key_points_path):
    """Auto-detect the number of keypoints from a keypoints file or directory.
    
    Args:
        key_points_path: Path to a .npz file or a directory containing .npz files.
    
    Returns:
        int: Number of keypoints per frame (shape[1] of the keypoints array).
    """
    if key_points_path is None:
        print("Warning: key_points_path is None, falling back to max_num_kps=256")
        return 256
    
    try:
        # If it's a directory, find the first .npz file
        if os.path.isdir(key_points_path):
            npz_files = sorted([f for f in os.listdir(key_points_path) if f.endswith('.npz')])
            if not npz_files:
                print(f"Warning: No .npz files found in {key_points_path}, falling back to max_num_kps=256")
                return 256
            file_path = os.path.join(key_points_path, npz_files[0])
        else:
            file_path = key_points_path
        
        data = np.load(file_path, allow_pickle=True)
        if 'bodies_candidate' in data:
            bodies_data = data['bodies_candidate']
        else:
            keys = list(data.keys())
            bodies_data = data[keys[0]]
        num_kps = bodies_data.shape[1]
        print(f"Auto-detected num_kps={num_kps} from {file_path}")
        return num_kps
    except Exception as e:
        print(f"Warning: Failed to detect num_kps from {key_points_path}: {e}, falling back to 256")
        return 256


def parse_args():
    parser = argparse.ArgumentParser(description="WanVideo Inference")
    parser.add_argument("--sft_model_high", type=str, default=None, help="Path to high noise SFT model")
    parser.add_argument("--sft_model_low", type=str, default=None, help="Path to low noise SFT model")
    parser.add_argument("--control_video", type=str, default=None, help="Path to control video")
    parser.add_argument("--reference_image", type=str, default="data/examples/wan/reference_image_girl.png", help="Path to reference image")
    parser.add_argument("--key_points", type=str, default=None, help="Path to key points")
    parser.add_argument("--depth_keypoints", type=str, default=None, help="Path to depth keypoints .npz file (per-keypoint depth from DVD depth estimation)")
    parser.add_argument("--depth_keypoints2", type=str, default=None, help="Path to depth keypoints2 .npz file (full depth map [T,H,W] from DVD depth estimation)")
    parser.add_argument("--input_image", type=str, default=None, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--height", type=int, default=None, help="Video height (default: read from control video resolution)")
    parser.add_argument("--width", type=int, default=None, help="Video width (default: read from control video resolution)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (default: read from control video length)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Output FPS")
    parser.add_argument("--use_usp", action="store_true", help="Use USP")
    parser.add_argument("--prompt", type=str, default="扁平风格动漫，一位长发少女优雅起舞。她五官精致，大眼睛明亮有神，黑色长发柔顺光泽。身穿淡蓝色T恤和深蓝色牛仔短裤。背景是粉色。", help="Prompt")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="Negative prompt")
    parser.add_argument("--control_video_3dpose", type=str, default=None, help="Path to 3D pose control video")
    parser.add_argument("--local_model_path", type=str, default="", help="Local path to model checkpoints")
    parser.add_argument("--ip2v", default=False, action="store_true", help="Whether to use ip2v pipeline.")
    parser.add_argument("--caption_csv", type=str, default=None, help="Path to caption CSV file (columns: path, text). When provided, prompt is read per-video from this file instead of --prompt.")
    parser.add_argument("--eval_output_suffix", type=str, default="evaluation", help="Subdirectory name for evaluation output (default: 'evaluation'). Use 'evaluation_wocaption' for no-caption mode.")
    parser.add_argument("--video_suffix", type=str, default="", help="Suffix appended to output video filename (before .mp4), e.g. ")
    parser.add_argument("--eval_limit", type=int, default=None, help="When set, only test the first N videos in batch/evaluation mode (default: None = test all)")
    parser.add_argument("--node_rank", type=int, default=0, help="Node index for multi-node batch inference (0-indexed). Used with --num_nodes to shard control_files across nodes.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes in multi-node batch inference. When >1, each node processes control_files[node_rank::num_nodes].")
    # Chunked inference: when enabled, split control video into overlapping temporal chunks and run pipeline per-chunk,
    # then concatenate outputs (dropping the first --chunk_overlap frames of every chunk after the first).
    parser.add_argument("--chunk_inference", default=False, action="store_true", help="Enable chunked inference (V1): split control video into overlapping chunks along temporal axis.")
    parser.add_argument("--chunk_inference_v2", default=False, action="store_true", help="Enable chunked inference V2: zero-overlap head-to-tail concatenation; each chunk = chunk_size frames, last frame of previous chunk becomes input_image of next chunk; the last chunk may be shorter than chunk_size and is snapped down to the nearest 4k+1 length. Mutually exclusive with --chunk_inference; takes precedence if both are set.")
    parser.add_argument("--chunk_size", type=int, default=81, help="Frames per chunk (should satisfy 4k+1 alignment, default 81).")
    parser.add_argument("--chunk_overlap", type=int, default=4, help="Temporal overlap (in frames) between adjacent chunks (default 4, V1 only). Stride = chunk_size - chunk_overlap.")

    # Pipeline kwargs group: all args in this group will be auto-forwarded to inputs_shared in wan_video.py
    # To add a new pipeline kwarg, simply add it here — no changes needed in wan_video.py.
    pipe_kwargs = parser.add_argument_group("pipeline_kwargs", "Args auto-forwarded to pipeline inputs_shared")
    pipe_kwargs.add_argument("--temporal_concat", default=False, action="store_true", help="Whether to use temporal concat instead of channel concat.")
    pipe_kwargs.add_argument("--first_as_guidance", default=False, action="store_true", help="Whether to use first frame as guidance.")
    pipe_kwargs.add_argument("--first_as_guidance_middle", default=False, action="store_true", help="Whether to use first frame as guidance in the middle of the video.")
    pipe_kwargs.add_argument("--first_as_guidance_adaptive", default=False, action="store_true", help="Whether to use first frame as guidance with adaptive weights on first/middle/last blocks.")
    pipe_kwargs.add_argument("--first_as_guidance_cross_attn", default=False, action="store_true", help="Whether to use first frame as guidance via concat self-attention: concat first-frame clean latent tokens with all tokens for self-attention, then remove them.")
    pipe_kwargs.add_argument("--attention_warp", default=False, action="store_true", help="Whether to use attention-based warp.")
    pipe_kwargs.add_argument("--fix_missing_warp", default=False, action="store_true", help="Whether to use fix-missing warp (V1): modify mask for missing keypoints and add keypoint index embedding.")
    pipe_kwargs.add_argument("--fix_missing_warp_v2", default=False, action="store_true", help="Whether to use fix-missing warp V2: keep original 0/1 mask, fill missing keypoint positions with 16ch learned embedding + 4ch keypoint index embedding concatenated to y.")
    pipe_kwargs.add_argument("--fix_missing_warp_v3", default=False, action="store_true", help="Whether to use fix-missing warp V3: same as V2 (mask 0/1, missing filled with 16ch learned embedding) but WITHOUT 4ch keypoint index embedding. No patch_embedding expansion needed.")
    pipe_kwargs.add_argument("--score_filter", default=False, action="store_true", help="Whether to apply score-based filtering on keypoints before warp (body>=0.3, hand>=0.3, face>=0.3, skip face jaw/nose bridge). Fully independent flag that can combine with any of the fix_missing_warp variants, or be used alone with plain DirectWarp.")
    pipe_kwargs.add_argument("--attention_warp_vis", default=False, action="store_true", help="Whether to visualize attention warp results via PCA.")
    pipe_kwargs.add_argument("--attention_warp_vis_dir", type=str, default="./output/attention_warp_vis", help="Directory to save attention warp PCA visualizations.")
    pipe_kwargs.add_argument("--latent_warp_vis", default=False, action="store_true", help="Whether to visualize latent-warp pipeline's image latent (before/after dit.patch_embedding projection) via PCA. Only effective when a latent warp pipeline is active (e.g. LOAD_POSE3_KEY_POINTS).")
    pipe_kwargs.add_argument("--latent_warp_vis_dir", type=str, default="./output/latent_warp_vis", help="Directory to save latent warp PCA visualizations.")
    pipe_kwargs.add_argument("--first_frame_attn_vis", default=False, action="store_true", help="Visualize how other frames' latents influence the first-frame latent via self-attention. For every denoising step (on the positive branch), compute softmax(Q @ K^T / sqrt(d)) from first-frame query tokens to all key tokens and save per-block heatmaps (grid over frames + per-frame summary bar chart + raw .npz). Only works without USP (run with NPROC_PER_NODE=1). Much slower than normal inference.")
    pipe_kwargs.add_argument("--first_frame_attn_vis_dir", type=str, default="./output/first_frame_attn_vis", help="Directory to save first-frame self-attention visualizations.")
    pipe_kwargs.add_argument("--first_frame_attn_vis_layers", type=str, default="", help="Comma-separated DiT block indices to record for first_frame_attn_vis (e.g. '0,10,20,29'). Empty (default) records ALL blocks, which may produce many images; prefer selecting a subset to reduce I/O.")
    pipe_kwargs.add_argument("--warp_limbs", default=False, action="store_true", help="Whether to also warp along pose limb connections (body/hand skeleton lines) in addition to keypoint positions.")
    pipe_kwargs.add_argument("--face_skip", default=False, action="store_true", help="Whether to skip drawing/warping FACE_SKIP_IDX keypoints (jaw contour 0-16 and nose bridge 27-35) to match pose visualization behavior. When enabled, these face keypoints are zeroed out inside both score-based filtering and the warp pipeline. Default False; the Shell launcher sets its default to follow WARP_LIMBS to preserve legacy behavior.")
    pipe_kwargs.add_argument("--vis_warp_keypoints", default=False, action="store_true", help="Whether to generate pose-style keypoints warp visualization video.")
    pipe_kwargs.add_argument("--vis_warp_keypoints_path", type=str, default="", help="(Auto-set) Output path for keypoints warp visualization video.")
    pipe_kwargs.add_argument("--token_replace", default=False, action="store_true", help="TOKEN_REPLACE (inference side): when combined with --chunk_inference_v2, the FIRST chunk runs normally with token_replace=False and a single-frame input_image; every SUBSEQUENT chunk runs with token_replace=True and uses the LAST 5 FRAMES of the previous chunk as input_image_frames. The chunk's control_video / key_points / depth_keypoints / depth_keypoints2 windows are shifted 4 pixel-frames earlier so that the 5 guidance frames align with the first 5 pose/keypoints/depth slots of the chunk (matching the training semantics 'first 5 pixel frames = temporal guidance'). The first 5 generated frames of such subsequent chunks are then dropped during concatenation (they reproduce already-emitted content), yielding an effective stride of chunk_size-5 frames.")

    args = parser.parse_args()

    # Record which keys belong to pipeline_kwargs group for auto-forwarding
    args.pipeline_kwargs_keys = [action.dest for action in pipe_kwargs._group_actions]

    return args

def load_caption_csv(csv_path):
    """Load caption CSV and return a dict mapping video relative path (stem) to prompt text.
    
    CSV format: path,text
    e.g.: video/000.mp4,"The video is ..."
    
    Returns a dict like {"000": "The video is ...", "001": "...", ...}
    """
    caption_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row["path"].strip()
            text = row["text"].strip()
            # Extract stem from path (e.g., "video/000.mp4" -> "000")
            stem = os.path.splitext(os.path.basename(video_path))[0]
            caption_map[stem] = text
    print(f"Loaded {len(caption_map)} captions from {csv_path}")
    return caption_map


args = parse_args()

# Load caption CSV if provided
caption_map = None
if args.caption_csv is not None:
    caption_map = load_caption_csv(args.caption_csv)

# Only compute default output path for single-file mode (reference_image is a file, not a directory)
if not os.path.isdir(args.reference_image):
    if args.output is None:
        if args.sft_model_high is None and args.sft_model_low is None:
            exp_name = "Wan2.2-I2V-A14B"
        else:
            model_path = args.sft_model_high if args.sft_model_high is not None else args.sft_model_low
            model_path = os.path.abspath(model_path)
            folder_name = os.path.basename(os.path.dirname(model_path))
            if folder_name in ["high_noise_model", "low_noise_model"]:
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            file_name = os.path.basename(model_path)
            file_name = file_name.split(".")[0]
            dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(model_path))))
            exp_name = f"{dataset_name}{os.sep}{folder_name}{os.sep}{file_name}"
        video_name = os.path.basename(args.control_video) if args.control_video is not None else os.path.basename(args.control_video_3dpose)
        assert args.control_video is not None or args.control_video_3dpose is not None
        ref_name = os.path.splitext(os.path.basename(args.reference_image))[0]
        file_name_without_ext, ext = os.path.splitext(video_name)
    num_frames_str = str(args.num_frames) if args.num_frames is not None else "auto"
    width_str = str(args.width) if args.width is not None else "auto"
    height_str = str(args.height) if args.height is not None else "auto"
    video_name = f"{file_name_without_ext}_{ref_name}_{width_str}x{height_str}_f{num_frames_str}{ext}"
    if args.use_usp:
        file_name_without_ext, ext = os.path.splitext(video_name)
        video_name = f"{file_name_without_ext}_useusp{ext}"
    if getattr(args, "chunk_inference_v2", False):
        file_name_without_ext, ext = os.path.splitext(video_name)
        video_name = f"{file_name_without_ext}_chunkv2{args.chunk_size}{ext}"
    elif getattr(args, "chunk_inference", False):
        file_name_without_ext, ext = os.path.splitext(video_name)
        video_name = f"{file_name_without_ext}_chunk{args.chunk_size}o{args.chunk_overlap}{ext}"
    if args.video_suffix:
        file_name_without_ext, ext = os.path.splitext(video_name)
        video_name = f"{file_name_without_ext}{args.video_suffix}{ext}"
    args.output = os.path.join("output", exp_name, video_name)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check if output file already exists, skip if so
    if os.path.exists(args.output):
        print(f"Output file already exists: {args.output}, skipping.")
        exit(0)
    
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", local_model_path=args.local_model_path),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", local_model_path=args.local_model_path),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", local_model_path=args.local_model_path),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", local_model_path=args.local_model_path),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/", local_model_path=args.local_model_path),
    redirect_common_files=False,
    use_usp=args.use_usp
)

if args.sft_model_high is not None:
    ### init dit.has_ref_conv
    # pipe.dit.has_ref_conv = True
    # pipe.dit.ref_conv = torch.nn.Conv2d(16, pipe.dit.dim, kernel_size=(2, 2), stride=(2, 2)).to(device=next(pipe.dit.parameters()).device, dtype=next(pipe.dit.parameters()).dtype)

    ### support ip2v pipeline 
    if getattr(args, "ip2v", False):
        pipe.dit.in_dim = 52
        pipe.dit.patch_embedding = torch.nn.Conv3d(
            pipe.dit.in_dim, pipe.dit.dim, kernel_size=pipe.dit.patch_size, stride=pipe.dit.patch_size,
                bias=pipe.dit.patch_embedding.bias is not None).to(device=pipe.device, dtype=pipe.torch_dtype)

    # fix_missing_warp V1 / V2 need patch_embedding expansion (+4ch for kp index embedding).
    # V3 keeps the original 20ch layout, so NO expansion is performed.
    # NOTE: score_filter is an independent flag and does not participate here.
    if getattr(args, "fix_missing_warp", False) or getattr(args, "fix_missing_warp_v2", False):
        old_in_dim = pipe.dit.in_dim
        new_in_dim = old_in_dim + 4
        pipe.dit.in_dim = new_in_dim
        pipe.dit.patch_embedding = torch.nn.Conv3d(
            new_in_dim, pipe.dit.dim, kernel_size=pipe.dit.patch_size, stride=pipe.dit.patch_size,
            bias=pipe.dit.patch_embedding.bias is not None).to(device=pipe.device, dtype=pipe.torch_dtype)
        variant = "v2" if getattr(args, "fix_missing_warp_v2", False) else "v1"
        print(f"Initialized fix_missing_warp ({variant}) for dit: expanded patch_embedding in_dim from {old_in_dim} to {new_in_dim}")

    if getattr(args, "fix_missing_warp_v2", False):
        # V2: need both kp_index_embedding_16ch (fill missing in latent) and kp_index_embedding_4ch (concat extra channels).
        max_num_kps = _detect_num_kps_from_path(getattr(args, "key_points", None))
        pipe.dit.kp_index_embedding_16ch = torch.nn.Embedding(max_num_kps, 16).to(
            device=pipe.device, dtype=pipe.torch_dtype)
        pipe.dit.kp_index_embedding_4ch = torch.nn.Embedding(max_num_kps, 4).to(
            device=pipe.device, dtype=pipe.torch_dtype)
        print(f"Initialized fix_missing_warp_v2 for dit: kp_index_embedding_16ch ({max_num_kps} -> 16ch) and kp_index_embedding_4ch ({max_num_kps} -> 4ch)")

    if getattr(args, "fix_missing_warp_v3", False):
        # V3: only kp_index_embedding_16ch is needed (no patch_embedding expansion, no 4ch embedding).
        max_num_kps = _detect_num_kps_from_path(getattr(args, "key_points", None))
        pipe.dit.kp_index_embedding_16ch = torch.nn.Embedding(max_num_kps, 16).to(
            device=pipe.device, dtype=pipe.torch_dtype)
        print(f"Initialized fix_missing_warp_v3 for dit: kp_index_embedding_16ch ({max_num_kps} -> 16ch), no patch_embedding expansion")

    ### change channel concat to temporal concat
    if getattr(args, "temporal_concat", False):
        pose_input_dim = 4 + 16
        pipe.dit.patch_embedding_pose = torch.nn.Conv3d(
            pose_input_dim, pipe.dit.dim, kernel_size=pipe.dit.patch_size, stride=pipe.dit.patch_size).to(device=next(pipe.dit.parameters()).device, dtype=next(pipe.dit.parameters()).dtype)
        print("Initialized temporal concat pipeline successfully: self.pipe.dit.patch_embedding_pose")



    ### init ref_detail_transfer layers: dedicated Q/K/V/O projections with zero init
    ref_detail_transfer_layers = getattr(args, "ref_detail_transfer_layers", None)
    if getattr(args, "first_as_guidance", False):
        num_blocks = len(pipe.dit.blocks)
        dim = pipe.dit.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        if ref_detail_transfer_layers is None:
            # Default: apply to last 1/8 of layers (deeper layers carry more detail),
            # consistent with wan_video.py inference default.
            ref_detail_transfer_layers = list(range(num_blocks * 7 // 8, num_blocks))
        elif isinstance(ref_detail_transfer_layers, str):
            ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
        for block_id in ref_detail_transfer_layers:
            block = pipe.dit.blocks[block_id]
            for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                setattr(block, name, layer)
        print(f"Initialized ref_detail_transfer layers (zero-init) for blocks: {ref_detail_transfer_layers}")


    if getattr(args, "first_as_guidance_middle", False):
        num_blocks = len(pipe.dit.blocks)
        dim = pipe.dit.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        if ref_detail_transfer_layers is None:
            # Default: apply to middle 2 layers, consistent with train.py
            ref_detail_transfer_layers = [num_blocks//2, num_blocks//2+1]
        elif isinstance(ref_detail_transfer_layers, str):
            ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
        for block_id in ref_detail_transfer_layers:
            block = pipe.dit.blocks[block_id]
            for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                setattr(block, name, layer)
        print(f"Initialized first_as_guidance_middle ref_detail_transfer layers (zero-init) for blocks: {ref_detail_transfer_layers}")


    if getattr(args, "first_as_guidance_adaptive", False):
        num_blocks = len(pipe.dit.blocks)
        dim = pipe.dit.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        if ref_detail_transfer_layers is None:
            ref_detail_transfer_layers = [0, num_blocks // 2, num_blocks - 1]
        elif isinstance(ref_detail_transfer_layers, str):
            ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
        for block_id in ref_detail_transfer_layers:
            block = pipe.dit.blocks[block_id]
            for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                setattr(block, name, layer)
            adaptive_weight = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
            block.ref_detail_adaptive_weight = adaptive_weight
        print(f"Initialized first_as_guidance_adaptive ref_detail_transfer layers (zero-init + adaptive weight) for blocks: {ref_detail_transfer_layers}")


    if getattr(args, "first_as_guidance_cross_attn", False):
        num_blocks = len(pipe.dit.blocks)
        cross_attn_layers = getattr(args, "ref_detail_transfer_layers", None)
        if cross_attn_layers is None:
            cross_attn_layers = list(range(num_blocks * 7 // 8, num_blocks))
        elif isinstance(cross_attn_layers, str):
            cross_attn_layers = [int(x) for x in cross_attn_layers.split(",")]
        print(f"Enabled first_as_guidance_cross_attn (concat self-attn, no gate) for blocks: {cross_attn_layers}")


    if getattr(args, "attention_warp", False):
        dim = pipe.dit.dim
        patch_size = pipe.dit.patch_size
        device = pipe.device
        dtype = pipe.torch_dtype

        # Dedicated patchify layers for control latent (16ch)
        pipe.dit.attn_warp_control_patchify = torch.nn.Conv3d(
            16, 16, kernel_size=(1,1,1), stride=(1,1,1)
        ).to(device=device, dtype=dtype)

        # Global Q/K projections for attention warp, executed once before DiT blocks
        for name in ['attn_warp_q', 'attn_warp_k']:
            layer = torch.nn.Linear(16, 16).to(device=device, dtype=dtype)
            setattr(pipe.dit, name, layer)

        # RMSNorm for Q and K (matching SelfAttention norm_q/norm_k)
        from diffsynth.models.wan_video_dit import RMSNorm
        pipe.dit.attn_warp_norm_q = RMSNorm(16).to(device=device, dtype=dtype)
        pipe.dit.attn_warp_norm_k = RMSNorm(16).to(device=device, dtype=dtype)

        print(f"Initialized attention_warp layers: patchify + global Q/K/norm_q/norm_k")


    # Initialize depth_embedding if depth_keypoints2 is provided (like first_as_guidance_middle pattern)
    if args.depth_keypoints2 is not None:
        dim = pipe.dit.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        depth_levels = pipe.dit.depth_levels  # 64
        pipe.dit.depth_embedding = torch.nn.Embedding(depth_levels, dim).to(device=device, dtype=dtype)
        print(f"Initialized depth_embedding for dit: {depth_levels} levels, dim={dim}")

    # TOKEN_REPLACE: create `patch_embedding_token_replace` as a clone of the current
    # `patch_embedding` (same in/out channels / kernel / stride / bias config). We
    # initialize it BEFORE `pipe.dit.load_state_dict(...)` so that when a SFT ckpt
    # produced by the TOKEN_REPLACE training run (which carries both the frozen
    # `patch_embedding.*` and the trainable `patch_embedding_token_replace.*` keys)
    # is loaded in strict mode, all keys line up. If the SFT ckpt does NOT carry
    # `patch_embedding_token_replace.*` keys, strict loading would fail -- user is
    # expected to only set --token_replace when evaluating a matching checkpoint.
    if getattr(args, "token_replace", False):
        _orig_pe = pipe.dit.patch_embedding
        pipe.dit.patch_embedding_token_replace = torch.nn.Conv3d(
            _orig_pe.in_channels, _orig_pe.out_channels,
            kernel_size=_orig_pe.kernel_size, stride=_orig_pe.stride,
            bias=_orig_pe.bias is not None,
        ).to(device=pipe.device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            pipe.dit.patch_embedding_token_replace.weight.copy_(_orig_pe.weight)
            if _orig_pe.bias is not None:
                pipe.dit.patch_embedding_token_replace.bias.copy_(_orig_pe.bias)
        print(
            f"[token_replace] Initialized dit.patch_embedding_token_replace "
            f"(in={_orig_pe.in_channels}, out={_orig_pe.out_channels}, "
            f"kernel={_orig_pe.kernel_size}) as a copy of dit.patch_embedding."
        )

    state_dict = load_state_dict(args.sft_model_high)
    pipe.dit.load_state_dict(state_dict)
    print(f"Loaded high noise SFT model from {args.sft_model_high}")

if args.sft_model_low is not None:
    ### init dit.has_ref_conv
    # pipe.dit2.has_ref_conv = True
    # pipe.dit2.ref_conv = torch.nn.Conv2d(16, pipe.dit2.dim, kernel_size=(2, 2), stride=(2, 2)).to(device=next(pipe.dit2.parameters()).device, dtype=next(pipe.dit2.parameters()).dtype)

    ### support ip2v pipeline 
    if getattr(args, "ip2v", False):
        pipe.dit2.in_dim = 52
        pipe.dit2.patch_embedding = torch.nn.Conv3d(
            pipe.dit2.in_dim, pipe.dit2.dim, kernel_size=pipe.dit2.patch_size, stride=pipe.dit2.patch_size,
                bias=pipe.dit2.patch_embedding.bias is not None).to(device=pipe.device, dtype=pipe.torch_dtype)

    # fix_missing_warp V1 / V2 need patch_embedding expansion (+4ch for kp index embedding).
    # V3 keeps the original 20ch layout, so NO expansion is performed.
    # NOTE: score_filter is an independent flag and does not participate here.
    if getattr(args, "fix_missing_warp", False) or getattr(args, "fix_missing_warp_v2", False):
        old_in_dim = pipe.dit2.in_dim
        new_in_dim = old_in_dim + 4
        pipe.dit2.in_dim = new_in_dim
        pipe.dit2.patch_embedding = torch.nn.Conv3d(
            new_in_dim, pipe.dit2.dim, kernel_size=pipe.dit2.patch_size, stride=pipe.dit2.patch_size,
            bias=pipe.dit2.patch_embedding.bias is not None).to(device=pipe.device, dtype=pipe.torch_dtype)
        variant = "v2" if getattr(args, "fix_missing_warp_v2", False) else "v1"
        print(f"Initialized fix_missing_warp ({variant}) for dit2: expanded patch_embedding in_dim from {old_in_dim} to {new_in_dim}")

    if getattr(args, "fix_missing_warp_v2", False):
        max_num_kps = _detect_num_kps_from_path(getattr(args, "key_points", None))
        pipe.dit2.kp_index_embedding_16ch = torch.nn.Embedding(max_num_kps, 16).to(
            device=pipe.device, dtype=pipe.torch_dtype)
        pipe.dit2.kp_index_embedding_4ch = torch.nn.Embedding(max_num_kps, 4).to(
            device=pipe.device, dtype=pipe.torch_dtype)
        print(f"Initialized fix_missing_warp_v2 for dit2: kp_index_embedding_16ch ({max_num_kps} -> 16ch) and kp_index_embedding_4ch ({max_num_kps} -> 4ch)")

    if getattr(args, "fix_missing_warp_v3", False):
        max_num_kps = _detect_num_kps_from_path(getattr(args, "key_points", None))
        pipe.dit2.kp_index_embedding_16ch = torch.nn.Embedding(max_num_kps, 16).to(
            device=pipe.device, dtype=pipe.torch_dtype)
        print(f"Initialized fix_missing_warp_v3 for dit2: kp_index_embedding_16ch ({max_num_kps} -> 16ch), no patch_embedding expansion")

    ### change channel concat to temporal concat
    if getattr(args, "temporal_concat", False):
        pose_input_dim = 4 + 16
        pipe.dit2.patch_embedding_pose = torch.nn.Conv3d(
            pose_input_dim, pipe.dit2.dim, kernel_size=pipe.dit2.patch_size, stride=pipe.dit2.patch_size).to(device=next(pipe.dit2.parameters()).device, dtype=next(pipe.dit2.parameters()).dtype)
        print("Initialized temporal concat pipeline successfully: self.pipe.dit2.patch_embedding_pose")


    ### init ref_detail_transfer layers: dedicated Q/K/V/O projections with zero init
    ref_detail_transfer_layers = getattr(args, "ref_detail_transfer_layers", None)
    if getattr(args, "first_as_guidance", False):
        num_blocks = len(pipe.dit2.blocks)
        dim = pipe.dit2.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        if ref_detail_transfer_layers is None:
            # Default: apply to last 1/8 of layers (deeper layers carry more detail),
            # consistent with wan_video.py inference default.
            ref_detail_transfer_layers = list(range(num_blocks * 7 // 8, num_blocks))
        elif isinstance(ref_detail_transfer_layers, str):
            ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
        for block_id in ref_detail_transfer_layers:
            block = pipe.dit2.blocks[block_id]
            for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                setattr(block, name, layer)
        print(f"Initialized ref_detail_transfer layers (zero-init) for blocks: {ref_detail_transfer_layers}")


    if getattr(args, "first_as_guidance_middle", False):
        num_blocks = len(pipe.dit2.blocks)
        dim = pipe.dit2.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        if ref_detail_transfer_layers is None:
            # Default: apply to middle 2 layers, consistent with train.py
            ref_detail_transfer_layers = [num_blocks//2, num_blocks//2+1]
        elif isinstance(ref_detail_transfer_layers, str):
            ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
        for block_id in ref_detail_transfer_layers:
            block = pipe.dit2.blocks[block_id]
            for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                setattr(block, name, layer)
        print(f"Initialized first_as_guidance_middle ref_detail_transfer layers (zero-init) for dit2 blocks: {ref_detail_transfer_layers}")


    if getattr(args, "first_as_guidance_adaptive", False):
        num_blocks = len(pipe.dit2.blocks)
        dim = pipe.dit2.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        if ref_detail_transfer_layers is None:
            ref_detail_transfer_layers = [0, num_blocks // 2, num_blocks - 1]
        elif isinstance(ref_detail_transfer_layers, str):
            ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
        for block_id in ref_detail_transfer_layers:
            block = pipe.dit2.blocks[block_id]
            for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                setattr(block, name, layer)
            adaptive_weight = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
            block.ref_detail_adaptive_weight = adaptive_weight
        print(f"Initialized first_as_guidance_adaptive ref_detail_transfer layers (zero-init + adaptive weight) for dit2 blocks: {ref_detail_transfer_layers}")


    if getattr(args, "first_as_guidance_cross_attn", False):
        num_blocks = len(pipe.dit2.blocks)
        cross_attn_layers = getattr(args, "ref_detail_transfer_layers", None)
        if cross_attn_layers is None:
            cross_attn_layers = list(range(num_blocks * 7 // 8, num_blocks))
        elif isinstance(cross_attn_layers, str):
            cross_attn_layers = [int(x) for x in cross_attn_layers.split(",")]
        print(f"Enabled first_as_guidance_cross_attn (concat self-attn, no gate) for dit2 blocks: {cross_attn_layers}")


    if getattr(args, "attention_warp", False):
        dim = pipe.dit2.dim
        patch_size = pipe.dit2.patch_size
        device = pipe.device
        dtype = pipe.torch_dtype

        # Dedicated patchify layers for control latent (16ch)
        pipe.dit2.attn_warp_control_patchify = torch.nn.Conv3d(
            16, 16, kernel_size=(1,1,1), stride=(1,1,1)
        ).to(device=device, dtype=dtype)

        # Global Q/K projections for attention warp, executed once before DiT blocks
        for name in ['attn_warp_q', 'attn_warp_k']:
            layer = torch.nn.Linear(16, 16).to(device=device, dtype=dtype)
            setattr(pipe.dit2, name, layer)

        # RMSNorm for Q and K (matching SelfAttention norm_q/norm_k)
        from diffsynth.models.wan_video_dit import RMSNorm
        pipe.dit2.attn_warp_norm_q = RMSNorm(16).to(device=device, dtype=dtype)
        pipe.dit2.attn_warp_norm_k = RMSNorm(16).to(device=device, dtype=dtype)

        print(f"Initialized attention_warp layers for dit2: patchify + global Q/K/norm_q/norm_k")


    # Initialize depth_embedding if depth_keypoints2 is provided (like first_as_guidance_middle pattern)
    if args.depth_keypoints2 is not None:
        dim = pipe.dit2.dim
        device = pipe.device
        dtype = pipe.torch_dtype
        depth_levels = pipe.dit2.depth_levels  # 64
        pipe.dit2.depth_embedding = torch.nn.Embedding(depth_levels, dim).to(device=device, dtype=dtype)
        print(f"Initialized depth_embedding for dit2: {depth_levels} levels, dim={dim}")

    # TOKEN_REPLACE: mirror the dit-side setup on dit2 (low noise model), so the
    # low-noise SFT ckpt with `patch_embedding_token_replace.*` keys also loads
    # cleanly in strict mode.
    if getattr(args, "token_replace", False):
        _orig_pe2 = pipe.dit2.patch_embedding
        pipe.dit2.patch_embedding_token_replace = torch.nn.Conv3d(
            _orig_pe2.in_channels, _orig_pe2.out_channels,
            kernel_size=_orig_pe2.kernel_size, stride=_orig_pe2.stride,
            bias=_orig_pe2.bias is not None,
        ).to(device=pipe.device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            pipe.dit2.patch_embedding_token_replace.weight.copy_(_orig_pe2.weight)
            if _orig_pe2.bias is not None:
                pipe.dit2.patch_embedding_token_replace.bias.copy_(_orig_pe2.bias)
        print(
            f"[token_replace] Initialized dit2.patch_embedding_token_replace "
            f"(in={_orig_pe2.in_channels}, out={_orig_pe2.out_channels}, "
            f"kernel={_orig_pe2.kernel_size}) as a copy of dit2.patch_embedding."
        )

    state_dict = load_state_dict(args.sft_model_low)
    pipe.dit2.load_state_dict(state_dict)
    print(f"Loaded low noise SFT model from {args.sft_model_low}")

# Only download example assets if they are actually referenced and not yet present.
# In batch mode (reference_image is a directory) these files are not used at all.
_example_files = [
    "data/examples/wan/control_video.mp4",
    "data/examples/wan/reference_image_girl.png",
]
_needs_download = (
    (not os.path.isdir(args.reference_image))
    and any(not os.path.exists(p) for p in _example_files)
    and (
        args.reference_image in _example_files
        or (args.control_video is not None and args.control_video in _example_files)
    )
)
if _needs_download:
    dataset_snapshot_download(
        dataset_id="DiffSynth-Studio/examples_in_diffsynth",
        local_dir="./",
        allow_file_pattern=_example_files,
    )
else:
    print("Skip dataset_snapshot_download: example assets not needed or already present.")

def create_grid_video(control_video_path, vis_warp_kps_path, reference_image_path, generated_video_path, grid_output_path, fps=15):
    """Create a 2x2 grid video combining control video, warped keypoints vis, reference image, and generated video.

    Layout:
        +-------------------+-------------------+
        | control video     | warped kps vis    |
        +-------------------+-------------------+
        | reference image   | generated video   |
        +-------------------+-------------------+

    All four quadrants are resized to the same resolution (matching the generated video).
    """
    import cv2
    import imageio

    # Read generated video to determine target resolution and frame count
    gen_reader = imageio.get_reader(generated_video_path)
    gen_frames = [f for f in gen_reader]
    gen_reader.close()
    if len(gen_frames) == 0:
        print("[create_grid_video] Generated video has no frames, skipping grid creation.")
        return
    target_h, target_w = gen_frames[0].shape[:2]
    num_frames = len(gen_frames)

    # Read control video frames
    ctrl_frames = []
    effective_ctrl_path = control_video_path
    if effective_ctrl_path is not None and os.path.exists(effective_ctrl_path):
        ctrl_reader = imageio.get_reader(effective_ctrl_path)
        ctrl_frames = [f for f in ctrl_reader]
        ctrl_reader.close()

    # Read warped keypoints visualization video frames
    vis_frames = []
    if vis_warp_kps_path is not None and os.path.exists(vis_warp_kps_path):
        vis_reader = imageio.get_reader(vis_warp_kps_path)
        vis_frames = [f for f in vis_reader]
        vis_reader.close()

    # Load reference image
    ref_img = np.array(Image.open(reference_image_path).resize((target_w, target_h), Image.LANCZOS))

    def get_frame(frame_list, idx, h, w):
        """Get a frame from list, pad with last frame or black if out of range, and resize."""
        if len(frame_list) == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        frame = frame_list[min(idx, len(frame_list) - 1)]
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        # Ensure 3 channels
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return frame

    # Build grid frames
    grid_frames = []
    for i in range(num_frames):
        top_left = get_frame(ctrl_frames, i, target_h, target_w)
        top_right = get_frame(vis_frames, i, target_h, target_w)
        bottom_left = ref_img.copy()
        bottom_right = gen_frames[i]
        if bottom_right.shape[:2] != (target_h, target_w):
            bottom_right = cv2.resize(bottom_right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        top_row = np.concatenate([top_left, top_right], axis=1)
        bottom_row = np.concatenate([bottom_left, bottom_right], axis=1)
        grid = np.concatenate([top_row, bottom_row], axis=0)
        grid_frames.append(grid)

    # Save grid video
    os.makedirs(os.path.dirname(grid_output_path) if os.path.dirname(grid_output_path) else ".", exist_ok=True)
    writer = imageio.get_writer(grid_output_path, fps=fps, quality=5)
    for frame in grid_frames:
        writer.append_data(frame)
    writer.close()
    print(f"[create_grid_video] Saved 2x2 grid video ({num_frames} frames, {target_w*2}x{target_h*2}) to {grid_output_path}")


def run_single_inference(pipe, args, reference_image_path, control_video_path, control_video_3dpose_path, key_points_path, output_path, save_frames=True, prompt_override=None, depth_keypoints_path=None, depth_keypoints2_path=None):
    """Run inference for a single (reference_image, control_video) pair."""
    # Check if output file already exists, skip if so
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}, skipping.")
        return

    # Determine height/width for this sample (may be read from control video)
    height = args.height
    width = args.width
    original_height = height
    original_width = width

    # Determine num_frames for this sample (may be clamped by video length)
    num_frames = args.num_frames  # Could be None, meaning "use control video length"
    original_num_frames = None  # Track original frame count before 4k+1 padding
    output_fps = args.fps # Default fps

    # Control video
    control_video = None
    if control_video_path is not None:
        # First load without resize to get original resolution if height/width not specified
        if height is None or width is None:
            temp_video = VideoData(control_video_path)
            first_frame = temp_video.data[0]
            orig_w, orig_h = first_frame.size
            if height is None:
                height = orig_h
                original_height = orig_h
            if width is None:
                width = orig_w
                original_width = orig_w
            print(f"height/width not specified, using control video resolution: {width}x{height}")
            
            # Get fps from control video
            video_fps = temp_video.get_fps()
            if video_fps is not None:
                output_fps = video_fps
                print(f"Using control video fps: {output_fps}")
            
            del temp_video
        else:
            # If height/width are specified, we still need to get fps
            temp_video = VideoData(control_video_path)
            video_fps = temp_video.get_fps()
            if video_fps is not None:
                output_fps = video_fps
                print(f"Using control video fps: {output_fps}")
            del temp_video

        # Ensure resolution is divisible by 16
        if height % 16 != 0 or width % 16 != 0:
            height = (height // 16) * 16
            width = (width // 16) * 16
            print(f"Adjusted resolution to {width}x{height} (divisible by 16)")

        control_video = VideoData(control_video_path, height=height, width=width)
        actual_length = len(control_video.data)
        if num_frames is None:
            # Default: use control video length
            num_frames = actual_length
            print(f"num_frames not specified, using control video length={num_frames}")
        elif num_frames > actual_length:
            # Use actual video length instead of requested num_frames
            num_frames = actual_length
            print(f"Warning: requested num_frames={args.num_frames} exceeds actual video length={actual_length}, using actual length={num_frames}")
        # Pad to 4k+1 if needed: copy last frame to satisfy pipeline requirement
        if (num_frames - 1) % 4 != 0:
            original_num_frames = num_frames
            num_frames = ((num_frames - 1) // 4 + 1) * 4 + 1
            print(f"Padding control video from {original_num_frames} to {num_frames} frames (4k+1 alignment)")
        control_video.set_length(actual_length)  # Use actual length to avoid index out of bounds
        control_video_frames = control_video.raw_data()
        # Pad by repeating last frame if needed
        while len(control_video_frames) < num_frames:
            control_video_frames.append(control_video_frames[-1])
        # Truncate if num_frames < actual_length
        control_video = control_video_frames[:num_frames]

    # Load 3D pose control video
    control_video_3dpose = None
    if control_video_3dpose_path is not None:
        # Get original resolution from 3dpose video if height/width not specified
        if height is None or width is None:
            temp_video = VideoData(control_video_3dpose_path)
            first_frame = temp_video.data[0]
            orig_w, orig_h = first_frame.size
            if height is None:
                height = orig_h
                original_height = orig_h
            if width is None:
                width = orig_w
                original_width = orig_w
            print(f"height/width not specified, using 3dpose control video resolution: {width}x{height}")
            
            # Get fps from 3dpose control video
            video_fps = temp_video.get_fps()
            if video_fps is not None:
                output_fps = video_fps
                print(f"Using 3dpose control video fps: {output_fps}")
            
            del temp_video
        else:
            # If height/width are specified, we still need to get fps
            temp_video = VideoData(control_video_3dpose_path)
            video_fps = temp_video.get_fps()
            if video_fps is not None:
                output_fps = video_fps
                print(f"Using 3dpose control video fps: {output_fps}")
            del temp_video

        # Ensure resolution is divisible by 16
        if height % 16 != 0 or width % 16 != 0:
            height = (height // 16) * 16
            width = (width // 16) * 16
            print(f"Adjusted resolution to {width}x{height} (divisible by 16)")

        control_video_3dpose = VideoData(control_video_3dpose_path, height=height, width=width)
        actual_length_3dpose = len(control_video_3dpose.data)
        if num_frames is None:
            # Default: use 3dpose control video length
            num_frames = actual_length_3dpose
            print(f"num_frames not specified, using 3dpose control video length={num_frames}")
        elif num_frames > actual_length_3dpose:
            # Use actual video length instead of requested num_frames
            num_frames = actual_length_3dpose
            print(f"Warning: requested num_frames exceeds actual 3dpose video length={actual_length_3dpose}, using actual length={num_frames}")
        # Pad to 4k+1 if needed: copy last frame to satisfy pipeline requirement
        if (num_frames - 1) % 4 != 0:
            if original_num_frames is None:
                original_num_frames = num_frames
            num_frames = ((num_frames - 1) // 4 + 1) * 4 + 1
            print(f"Padding 3dpose control video from {original_num_frames} to {num_frames} frames (4k+1 alignment)")
        control_video_3dpose.set_length(actual_length_3dpose)  # Use actual length to avoid index out of bounds
        control_video_3dpose_frames = control_video_3dpose.raw_data()
        # Pad by repeating last frame if needed
        while len(control_video_3dpose_frames) < num_frames:
            control_video_3dpose_frames.append(control_video_3dpose_frames[-1])
        # Truncate if num_frames < actual_length
        control_video_3dpose = control_video_3dpose_frames[:num_frames]
        print(f"Loaded 3D pose control video from {control_video_3dpose_path}, num_frames: {len(control_video_3dpose)}")

    # control_video and control_video_3dpose only one of them is not None
    assert (control_video_path is None and control_video_3dpose_path is not None) or (control_video_path is not None and control_video_3dpose_path is None)

    reference_image_ori = Image.open(reference_image_path)
    reference_image = reference_image_ori.resize((width, height))
    print(f"reference_image_path: {reference_image_path}, original size: {reference_image_ori.size}, width: {width}, height: {height}")

    # Load key points (supports both 2D DWPose 18pts and 3D SMPL+DWPose 134pts)
    key_points = None
    if key_points_path is not None:
        # Use whichever control video path is available for resolution detection
        effective_control_video_path = control_video_path if control_video_path is not None else control_video_3dpose_path
        keypoints_loader = LoadKeypoints(
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            height=height,
            width=width,
            control_video_path=effective_control_video_path
        )
        key_points = keypoints_loader(key_points_path)
        print(f"Loaded key points from {key_points_path}, shape: {key_points.shape}")

    # Load depth keypoints (sparse per-keypoint depth)
    depth_keypoints = None
    if depth_keypoints_path is not None:
        depth_kp_loader = LoadDepthKeypoints(
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        depth_keypoints = depth_kp_loader(depth_keypoints_path)
        print(f"Loaded depth keypoints from {depth_keypoints_path}, shape: {depth_keypoints.shape}")

    # Load depth keypoints2 (full depth map [T, H, W])
    depth_keypoints2 = None
    if depth_keypoints2_path is not None:
        depth_kp2_loader = LoadDepthKeypoints2(
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            patch_size_t=4,
            patch_size_h=16,
            patch_size_w=16,
        )
        depth_keypoints2 = depth_kp2_loader(depth_keypoints2_path)
        print(f"Loaded depth keypoints2 from {depth_keypoints2_path}, shape: {depth_keypoints2.shape}")

    # If 3D pose control video is provided, use it as the control_video
    if control_video_3dpose is not None:
        pipe_control_video = control_video_3dpose
        print(f"Using 3D pose control video as control_video")
    else:
        pipe_control_video = control_video
        print(f"Using 2D pose control video as control_video")

    # Check resolution
    print(f"pipe_control_video.shape={np.shape(pipe_control_video[0])}, reference_image.shape={np.shape(reference_image)}")

    # Use per-video prompt if provided, otherwise fall back to args.prompt
    effective_prompt = prompt_override if prompt_override is not None else args.prompt
    # print(f"Using prompt: {effective_prompt[:100]}..." if len(effective_prompt) > 100 else f"Using prompt: {effective_prompt}")
    print(f"Using prompt: {effective_prompt}")

    # Set vis_warp_keypoints_path dynamically based on output_path
    if getattr(args, 'vis_warp_keypoints', False):
        vis_dir = os.path.dirname(output_path) or "."
        vis_stem = os.path.splitext(os.path.basename(output_path))[0]
        args.vis_warp_keypoints_path = os.path.join(vis_dir, f"{vis_stem}_vis_warp_kps.mp4")

    def _slice_temporal(arr, start, end):
        """Slice the temporal (first) axis of a list / ndarray / tensor between [start, end).
        Supports:
          - Python list of frames (control_video frames) -> returns sliced list.
          - torch.Tensor or np.ndarray with a leading temporal dimension.
          - None -> returns None.
        """
        if arr is None:
            return None
        if isinstance(arr, list):
            return arr[start:end]
        # Tensors / ndarrays use slicing on dim 0
        return arr[start:end]

    def _run_pipe_chunk(chunk_control_video, chunk_key_points, chunk_depth_kp,
                        chunk_depth_kp2, chunk_num_frames, chunk_ref_image,
                        chunk_input_image_frames=None, chunk_token_replace=False):
        """Run the pipeline once for a single chunk with the given sliced inputs.

        TOKEN_REPLACE handling: when ``chunk_token_replace`` is True, we temporarily
        override ``args.token_replace`` so the pipeline_kwargs auto-forwarding routes
        the correct boolean into ``inputs_shared`` for this specific chunk only. The
        previous value is restored after the call. ``chunk_input_image_frames`` is
        forwarded as a first-class pipeline argument consumed by
        WanVideoUnit_ImageEmbedderVAE when ``token_replace=True``.
        """
        _prev_tr = getattr(args, "token_replace", False)
        args.token_replace = bool(chunk_token_replace)
        try:
            return pipe(
                prompt=effective_prompt,
                negative_prompt=args.negative_prompt,
                input_image=chunk_ref_image,
                input_image_frames=chunk_input_image_frames,
                control_video=chunk_control_video,
                key_points=chunk_key_points,
                depth_keypoints=chunk_depth_kp,
                depth_keypoints2=chunk_depth_kp2,
                height=height, width=width, num_frames=chunk_num_frames,
                seed=args.seed, tiled=True,
                switch_DiT_boundary=0.9,
                args=args,
            )
        finally:
            args.token_replace = _prev_tr

    def _frame_to_pil(frame, target_w, target_h):
        """Convert a video frame (PIL/ndarray/tensor) to a PIL.Image at target size."""
        if isinstance(frame, Image.Image):
            pil = frame
        elif isinstance(frame, np.ndarray):
            arr = frame
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [1, 3, 4]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        elif isinstance(frame, torch.Tensor):
            arr = frame.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = frame  # best-effort fallback
        if isinstance(pil, Image.Image) and pil.size != (target_w, target_h):
            pil = pil.resize((target_w, target_h))
        return pil

    use_chunk_v2 = getattr(args, 'chunk_inference_v2', False)
    use_chunk = getattr(args, 'chunk_inference', False)
    if use_chunk_v2 and use_chunk:
        print("[chunk] Both --chunk_inference and --chunk_inference_v2 are set; V2 takes precedence.")
        use_chunk = False

    if use_chunk_v2:
        # ---- V2: 1-frame overlap, head-to-tail concatenation ----
        # Each chunk generates exactly `chunk_size` frames (except the last,
        # whose size is rounded UP to the smallest 4k+1 that covers all remaining
        # frames). Semantics:
        #   * The LAST frame of chunk i-1 is fed as the INPUT_IMAGE (first-frame
        #     anchor) of chunk i, so chunk i's frame 0 and chunk (i-1)'s last
        #     frame occupy the SAME global frame position.
        #   * When concatenating, chunk i (i>0) drops its first frame, since
        #     that frame was already emitted as the last frame of chunk i-1.
        # Stride = chunk_size - 1 frames (one-frame overlap by design).
        # Stride behavior depends on --token_replace (see --token_replace help text):
        #   * token_replace=False (default): stride = chunk_size - 1 (1-frame overlap).
        #     chunk i (i>0) uses the LAST frame of chunk i-1 as the single-frame
        #     input_image anchor (original V2 behavior).
        #   * token_replace=True: stride = chunk_size - 5 (5-frame overlap).
        #     chunk 0 runs with token_replace=False + single-frame input_image
        #     (normal first chunk). Every subsequent chunk i>0 runs with
        #     token_replace=True and uses the LAST 5 FRAMES of chunk i-1 as
        #     input_image_frames; the chunk's control_video / key_points /
        #     depth_keypoints / depth_keypoints2 windows are shifted so the 5
        #     guidance frames align with the first 5 pose/keypoints/depth slots
        #     of the chunk. The first 5 generated frames of such subsequent
        #     chunks reproduce already-emitted content and are dropped during
        #     concatenation.
        chunk_size = int(getattr(args, 'chunk_size', 81))
        overlap_frames = 5 if getattr(args, 'token_replace', False) else 1
        if overlap_frames == 5:
            # Need chunk_size > 5 AND 4k+1-aligned -> minimum 9 (==4*2+1).
            assert chunk_size >= 9, (
                f"chunk_size ({chunk_size}) must be >= 9 when --token_replace is on with "
                f"--chunk_inference_v2 (5 frames go to input_image_frames; remaining >= 4 frames "
                f"must be newly generated per chunk)"
            )
        else:
            assert chunk_size >= 5, f"chunk_size ({chunk_size}) must be >= 5 (smallest 4k+1)"
        # Enforce 4k+1 alignment on chunk_size (all middle chunks use this size).
        if (chunk_size - 1) % 4 != 0:
            aligned = ((chunk_size - 1) // 4 + 1) * 4 + 1
            print(f"[chunk-v2] chunk_size={chunk_size} is not 4k+1, aligning to {aligned}")
            chunk_size = aligned

        # Stride between adjacent chunks. Because chunk_size=4k+1, step is always a
        # multiple of 4 (overlap_frames in {1, 5} -> step in {4k, 4(k-1)} both 4k'-like),
        # which preserves pixel<->latent alignment for depth_keypoints2 slicing.
        step = chunk_size - overlap_frames

        def _ceil_4k1(n):
            """Round n UP to the smallest value of the form 4k+1 (k>=1, so >=5)."""
            if n <= 5:
                return 5
            # n = 4k + r (r in 0..3). We want the smallest 4k'+1 >= n.
            # Equivalently k' = ceil((n-1)/4).
            return ((n - 1 + 3) // 4) * 4 + 1

        def _ceil_4k1_tail(n, min_size):
            """Like _ceil_4k1 but ensures the result is at least ``min_size`` (also 4k+1).
            Used for subsequent (i>0) tail chunks when overlap_frames=5, so the tail
            is guaranteed to have MORE than 5 frames (i.e. >=9 = smallest 4k+1 >5),
            otherwise the post-chunk 5-frame drop would leave nothing usable.
            """
            result = _ceil_4k1(n)
            if result < min_size:
                result = min_size
            return result

        # Build chunk plan: list of (start, size) tuples.
        # chunk i covers global frames [start_i, start_i + size_i).
        # Adjacent chunks share exactly ``overlap_frames`` global frames:
        # start_{i+1} = start_i + size_i - overlap_frames.
        # Plan stops when the union of chunks covers [0, num_frames).
        chunk_plan = []
        s_cur = 0
        # Minimum tail size for i>0 chunks under token_replace: must be >5 AND 4k+1 -> 9.
        tail_min_size_subsequent = 9 if overlap_frames == 5 else 5
        while True:
            remaining_end = num_frames  # exclusive global end
            # If a chunk of size `chunk_size` starting at s_cur fully covers up to
            # remaining_end (i.e. s_cur + chunk_size >= remaining_end), we may shrink
            # this chunk to a smaller 4k+1 tail. Otherwise use a full-size chunk.
            if s_cur + chunk_size >= remaining_end:
                # Tail chunk: must cover [s_cur, num_frames). needed >= num_frames - s_cur.
                needed = remaining_end - s_cur
                if len(chunk_plan) == 0:
                    # Tail chunk is ALSO the first chunk -> no 5-frame drop constraint.
                    tail_size = _ceil_4k1(needed)
                else:
                    tail_size = _ceil_4k1_tail(needed, tail_min_size_subsequent)
                # Tail must not exceed chunk_size (since chunk_size is itself 4k+1 and
                # the loop ensures s_cur + chunk_size >= remaining_end => tail_size <= chunk_size).
                tail_size = min(tail_size, chunk_size)
                chunk_plan.append((s_cur, tail_size))
                break
            else:
                chunk_plan.append((s_cur, chunk_size))
                s_cur += step  # next chunk starts overlap_frames before this chunk's end

        # Planned global coverage end (exclusive) = last chunk's end. Due to
        # ceil-to-4k+1, it may be >= num_frames; we'll truncate after concatenation.
        last_start, last_size = chunk_plan[-1]
        planned_end = last_start + last_size

        # Total concatenated frame count: chunk 0 contributes size0, each
        # subsequent chunk contributes (size - overlap_frames) because we drop its
        # first ``overlap_frames`` frames.
        effective_total = chunk_plan[0][1] + sum(sz - overlap_frames for _, sz in chunk_plan[1:])
        print(f"[chunk-v2] chunk_inference_v2 enabled: total_frames={num_frames}, "
              f"chunk_size={chunk_size}, overlap_frames={overlap_frames}, step={step}, "
              f"token_replace={getattr(args, 'token_replace', False)}, "
              f"num_chunks={len(chunk_plan)}, "
              f"plan={chunk_plan}, planned_global_end={planned_end}, "
              f"effective_concat_frames={effective_total}")

        total_latent_frames_v2 = depth_keypoints2.shape[0] if depth_keypoints2 is not None else None

        # --- vis_warp_keypoints chunk support (V2) ---
        vis_kps_enabled = getattr(args, 'vis_warp_keypoints', False) and \
                          hasattr(args, 'vis_warp_keypoints_path') and \
                          bool(args.vis_warp_keypoints_path)
        final_vis_kps_path = args.vis_warp_keypoints_path if vis_kps_enabled else None
        chunk_vis_tmp_paths = []
        full_vis_frames = []

        full_video = []
        current_ref_image = reference_image
        # TOKEN_REPLACE (chunk-v2 mode): list of the last 5 PIL frames of the
        # previously generated chunk, to be passed as input_image_frames to the
        # next chunk (i>0). None for chunk 0.
        current_ref_frames = None
        for chunk_idx, (s, size) in enumerate(chunk_plan):
            e = s + size
            _is_tr_chunk = (getattr(args, 'token_replace', False) and chunk_idx > 0)
            print(f"\n[chunk-v2] === Running chunk {chunk_idx+1}/{len(chunk_plan)}: "
                  f"frames [{s}, {e}) size={size}, "
                  f"ref={'original' if chunk_idx == 0 else (f'prev_chunk_last_5_frames(@global_frames_{s}..{s+4})' if _is_tr_chunk else f'prev_chunk_last_frame(@global_frame_{s})')}, "
                  f"token_replace={_is_tr_chunk} ===")

            # Slice temporal inputs. For chunks whose end exceeds the original
            # num_frames (possible only for the tail chunk due to ceil-to-4k+1),
            # _slice_temporal naturally truncates at end; we then pad by repeating
            # the last slice element so chunk_ctrl has exactly `size` frames.
            chunk_ctrl = _slice_temporal(pipe_control_video, s, e)
            chunk_kp = _slice_temporal(key_points, s, e)
            chunk_dk = _slice_temporal(depth_keypoints, s, e)

            def _pad_to_size(arr, target_size):
                """Pad a list/tensor/ndarray along the first axis by repeating the
                last element until its length reaches `target_size`. Used when the
                tail chunk's range [s, s+size) extends beyond the input length."""
                if arr is None:
                    return None
                if isinstance(arr, list):
                    while len(arr) < target_size:
                        arr.append(arr[-1])
                    return arr
                # torch.Tensor / np.ndarray
                cur_len = arr.shape[0]
                if cur_len >= target_size:
                    return arr
                pad_count = target_size - cur_len
                # Repeat the last row pad_count times
                try:
                    import torch as _torch
                    if isinstance(arr, _torch.Tensor):
                        last = arr[-1:].expand(pad_count, *arr.shape[1:])
                        return _torch.cat([arr, last], dim=0)
                except Exception:
                    pass
                import numpy as _np
                last = _np.repeat(arr[-1:], pad_count, axis=0)
                return _np.concatenate([arr, last], axis=0)

            chunk_ctrl = _pad_to_size(chunk_ctrl, size)
            chunk_kp = _pad_to_size(chunk_kp, size)
            chunk_dk = _pad_to_size(chunk_dk, size)

            chunk_dk2 = None
            if depth_keypoints2 is not None:
                # V2 start s = i * (chunk_size - 1) may NOT be a multiple of 4 for
                # i > 0. Use nearest-neighbor latent-frame mapping.
                f_start = int(round(s / 4.0))
                f_len = (size - 1) // 4 + 1
                f_end = f_start + f_len
                if total_latent_frames_v2 is not None and f_end > total_latent_frames_v2:
                    f_end = total_latent_frames_v2
                    f_start = max(0, f_end - f_len)
                    print(f"[chunk-v2] Warning: depth_keypoints2 clamped to [{f_start}, {f_end})")
                chunk_dk2 = depth_keypoints2[f_start:f_end]
                # Pad to f_len (match what pipeline expects for `size` pixel frames)
                if chunk_dk2.shape[0] < f_len:
                    pad_count = f_len - chunk_dk2.shape[0]
                    try:
                        import torch as _torch
                        if isinstance(chunk_dk2, _torch.Tensor):
                            last = chunk_dk2[-1:].expand(pad_count, *chunk_dk2.shape[1:])
                            chunk_dk2 = _torch.cat([chunk_dk2, last], dim=0)
                    except Exception:
                        pass
                if s % 4 != 0:
                    print(f"[chunk-v2] Note: chunk start s={s} is not a multiple of 4; "
                          f"depth_keypoints2 sliced by nearest-neighbor latent frame "
                          f"[{f_start}, {f_end}) (may drift <= 2 pixel frames).")

            # Redirect vis_warp_keypoints output to a per-chunk temp file.
            if vis_kps_enabled:
                _vis_dir = os.path.dirname(final_vis_kps_path) or "."
                _vis_stem = os.path.splitext(os.path.basename(final_vis_kps_path))[0]
                chunk_vis_path = os.path.join(_vis_dir, f"{_vis_stem}_chunk{chunk_idx:03d}.mp4")
                args.vis_warp_keypoints_path = chunk_vis_path
                chunk_vis_tmp_paths.append(chunk_vis_path)

            chunk_video = _run_pipe_chunk(
                chunk_ctrl, chunk_kp, chunk_dk, chunk_dk2, size,
                current_ref_image,
                chunk_input_image_frames=current_ref_frames if _is_tr_chunk else None,
                chunk_token_replace=_is_tr_chunk,
            )
            chunk_video_list = list(chunk_video)

            # Collect vis frames with the same ``overlap_frames``-drop logic as main video.
            if vis_kps_enabled and chunk_vis_tmp_paths and os.path.exists(chunk_vis_tmp_paths[-1]):
                try:
                    import imageio as _imageio
                    _vis_reader = _imageio.get_reader(chunk_vis_tmp_paths[-1])
                    _chunk_vis_frames = [f for f in _vis_reader]
                    _vis_reader.close()
                    if chunk_idx == 0:
                        full_vis_frames.extend(_chunk_vis_frames)
                    else:
                        # Drop first ``overlap_frames`` frames (aliased to prev chunk's tail).
                        _drop = min(overlap_frames, len(_chunk_vis_frames))
                        full_vis_frames.extend(_chunk_vis_frames[_drop:])
                    print(f"[chunk-v2] Collected {len(_chunk_vis_frames)} vis frames from chunk {chunk_idx} "
                          f"(contributed {len(_chunk_vis_frames) if chunk_idx == 0 else max(0, len(_chunk_vis_frames)-overlap_frames)}, "
                          f"total so far: {len(full_vis_frames)})")
                except Exception as _e:
                    print(f"[chunk-v2] Warning: failed to read vis frames from {chunk_vis_tmp_paths[-1]}: {_e}")

            # Concatenate main video: chunk 0 keeps all frames; chunk i>0 drops
            # its first ``overlap_frames`` frames (aliased to chunk (i-1)'s tail).
            if chunk_idx == 0:
                full_video.extend(chunk_video_list)
            else:
                full_video.extend(chunk_video_list[overlap_frames:])

            # Next chunk's reference(s):
            #   * token_replace on AND there is a next chunk -> save LAST 5 frames for
            #     input_image_frames; input_image falls back to the last frame to keep
            #     WanVideoUnit_ImageEmbedderVAE / CLIPEmbedder happy with a non-None PIL.
            #   * token_replace off -> keep legacy single-last-frame behavior.
            if chunk_idx + 1 < len(chunk_plan):
                last_frame = chunk_video_list[-1]
                current_ref_image = _frame_to_pil(last_frame, width, height)
                if getattr(args, 'token_replace', False):
                    # Take the last 5 frames (pad by repeating the first available
                    # frame if the chunk is shorter than 5 -- should not happen
                    # under current plan, but is a safe guard).
                    _tail5 = chunk_video_list[-5:]
                    while len(_tail5) < 5:
                        _tail5 = [_tail5[0]] + _tail5
                    current_ref_frames = [_frame_to_pil(f, width, height) for f in _tail5]
                    print(f"[chunk-v2] Next chunk will use the LAST 5 frames of current chunk "
                          f"(global frames {s + size - 5}..{s + size - 1}) as input_image_frames "
                          f"(token_replace=True); input_image set to the last frame.")
                else:
                    current_ref_frames = None
                    print(f"[chunk-v2] Next chunk will use the LAST frame of current chunk "
                          f"(global frame {s + size - 1}) as reference_image (= next chunk's first frame)")

        # Truncate to num_frames in case the tail chunk's ceil-to-4k+1 made
        # the concatenated output longer than num_frames.
        if len(full_video) > num_frames:
            print(f"[chunk-v2] Truncating concatenated output from {len(full_video)} to {num_frames} frames "
                  f"(tail chunk over-generated {len(full_video) - num_frames} frames due to 4k+1 rounding)")
            full_video = full_video[:num_frames]
            if len(full_vis_frames) > num_frames:
                full_vis_frames = full_vis_frames[:num_frames]

        video = full_video
        print(f"[chunk-v2] Finished chunked inference V2: concatenated {len(video)} frames "
              f"from {len(chunk_plan)} chunks (target num_frames={num_frames})")

        # Merge per-chunk vis videos.
        if vis_kps_enabled and full_vis_frames:
            try:
                import imageio as _imageio
                os.makedirs(os.path.dirname(final_vis_kps_path) if os.path.dirname(final_vis_kps_path) else ".", exist_ok=True)
                _imageio.mimsave(final_vis_kps_path, full_vis_frames, fps=15)
                print(f"[chunk-v2] Merged {len(full_vis_frames)} vis frames → {final_vis_kps_path}")
            except Exception as _e:
                print(f"[chunk-v2] Warning: failed to write merged vis video: {_e}")
            for _tmp in chunk_vis_tmp_paths:
                try:
                    if os.path.exists(_tmp):
                        os.remove(_tmp)
                except Exception:
                    pass
            args.vis_warp_keypoints_path = final_vis_kps_path

        # Downstream "remove padding" block will trim to original_num_frames.
        # V2 output length equals num_frames here (by truncation above), so the
        # existing trim logic works without modification.

    elif use_chunk:
        chunk_size = int(getattr(args, 'chunk_size', 81))
        chunk_overlap = int(getattr(args, 'chunk_overlap', 4))
        assert chunk_size > chunk_overlap >= 0, \
            f"chunk_size ({chunk_size}) must be > chunk_overlap ({chunk_overlap}) >= 0"
        # Enforce 4k+1 alignment on chunk_size to satisfy pipeline requirement
        if (chunk_size - 1) % 4 != 0:
            aligned = ((chunk_size - 1) // 4 + 1) * 4 + 1
            print(f"[chunk] chunk_size={chunk_size} is not 4k+1, aligning to {aligned}")
            chunk_size = aligned

        # IMPORTANT: step (= chunk_size - effective_overlap) MUST be a multiple of 4
        # to keep pixel-frame chunk boundaries aligned with VAE temporal compression
        # (factor 4). Otherwise depth_keypoints2 (which lives in latent-frame space,
        # f = (T-1)//4 + 1) can NOT be sliced consistently with pipe_control_video
        # (pixel-frame space), and the latent first_frame_latents anchor (always the
        # 1st latent frame = pixel frames [0,4)) would drift across chunks.
        requested_step = chunk_size - chunk_overlap
        step = (requested_step // 4) * 4
        if step <= 0:
            step = 4  # minimum safe stride
        effective_overlap = chunk_size - step
        if step != requested_step:
            print(f"[chunk] step adjusted from {requested_step} to {step} (must be a multiple of 4 "
                  f"for VAE temporal alignment); effective_overlap = {effective_overlap}")

        # Compute chunk start indices so that the final chunk covers the last frame.
        # Note: num_frames here is already 4k+1-aligned (possibly padded from original_num_frames).
        # All starts are multiples of 4 (step is a multiple of 4), so the pixel→latent
        # mapping start // 4 is exact for every chunk.
        starts = []
        s = 0
        while s + chunk_size <= num_frames:
            starts.append(s)
            if s + chunk_size == num_frames:
                break
            s += step
        # If there are remaining frames not yet covered, add a tail chunk whose
        # start is also aligned to 4. The tail chunk may overlap the previous
        # chunk by MORE than effective_overlap (acceptable): its start is snapped
        # DOWN to the nearest multiple of 4 to preserve latent alignment.
        if len(starts) == 0:
            starts.append(0)
        elif starts[-1] + chunk_size < num_frames:
            tail_start = max(0, num_frames - chunk_size)
            tail_start = (tail_start // 4) * 4  # snap down to multiple of 4
            if tail_start != starts[-1]:
                starts.append(tail_start)

        # Final safety check: all starts must be multiples of 4.
        assert all(s % 4 == 0 for s in starts), \
            f"[chunk] internal error: chunk starts must be multiples of 4, got {starts}"

        print(f"[chunk] chunk_inference enabled: total_frames={num_frames}, "
              f"chunk_size={chunk_size}, requested_overlap={chunk_overlap}, "
              f"effective_overlap={effective_overlap}, step={step}, "
              f"num_chunks={len(starts)}, starts={starts}")

        # For depth_keypoints2 (shape [f, h, w] where f = (T-1)//4+1), the temporal
        # axis is in latent-frame units. Because every chunk start s is a multiple
        # of 4, pixel-frame range [s, s+chunk_size) corresponds EXACTLY to
        # latent-frame range [s//4, s//4 + latent_chunk_size).
        latent_chunk_size = (chunk_size - 1) // 4 + 1
        total_latent_frames = depth_keypoints2.shape[0] if depth_keypoints2 is not None else None

        # --- vis_warp_keypoints chunk support ---
        # Pipeline writes the warp-keypoints visualization to args.vis_warp_keypoints_path
        # on every pipe() call, overwriting the same file each time. In chunk mode we
        # redirect each chunk to a temporary per-chunk path, collect the frames with the
        # same overlap-drop logic used for the main video, and write the final merged
        # visualization once after all chunks are done.
        vis_kps_enabled = getattr(args, 'vis_warp_keypoints', False) and \
                          hasattr(args, 'vis_warp_keypoints_path') and \
                          bool(args.vis_warp_keypoints_path)
        final_vis_kps_path = args.vis_warp_keypoints_path if vis_kps_enabled else None
        chunk_vis_tmp_paths = []   # per-chunk temp paths (cleaned up after merge)
        full_vis_frames = []       # merged vis frames (pixel-domain, uint8 numpy)

        full_video = []
        # Initial reference image for the first chunk is the user-provided one.
        current_ref_image = reference_image
        for chunk_idx, s in enumerate(starts):
            e = s + chunk_size
            print(f"\n[chunk] === Running chunk {chunk_idx+1}/{len(starts)}: "
                  f"frames [{s}, {e}) (chunk_size={chunk_size}), "
                  f"ref={'original' if chunk_idx == 0 else f'pred@frame{s}'} ===")

            chunk_ctrl = _slice_temporal(pipe_control_video, s, e)
            chunk_kp = _slice_temporal(key_points, s, e)
            chunk_dk = _slice_temporal(depth_keypoints, s, e)
            chunk_dk2 = None
            if depth_keypoints2 is not None:
                f_start = s // 4
                f_end = f_start + latent_chunk_size
                # Clamp against total_latent_frames for safety (shouldn't happen after
                # 4k+1 alignment of num_frames, but be defensive).
                if total_latent_frames is not None and f_end > total_latent_frames:
                    f_end = total_latent_frames
                    f_start = max(0, f_end - latent_chunk_size)
                    print(f"[chunk] Warning: depth_keypoints2 clamped to [{f_start}, {f_end})")
                chunk_dk2 = depth_keypoints2[f_start:f_end]

            # Redirect vis_warp_keypoints output to a per-chunk temp file so that
            # successive chunks do not overwrite each other's visualization.
            if vis_kps_enabled:
                _vis_dir = os.path.dirname(final_vis_kps_path) or "."
                _vis_stem = os.path.splitext(os.path.basename(final_vis_kps_path))[0]
                chunk_vis_path = os.path.join(_vis_dir, f"{_vis_stem}_chunk{chunk_idx:03d}.mp4")
                args.vis_warp_keypoints_path = chunk_vis_path
                chunk_vis_tmp_paths.append(chunk_vis_path)

            chunk_video = _run_pipe_chunk(
                chunk_ctrl, chunk_kp, chunk_dk, chunk_dk2, chunk_size,
                current_ref_image,
            )

            # Collect vis frames with the SAME overlap-drop logic as the main video.
            if vis_kps_enabled and chunk_vis_tmp_paths and os.path.exists(chunk_vis_tmp_paths[-1]):
                try:
                    import imageio as _imageio
                    _vis_reader = _imageio.get_reader(chunk_vis_tmp_paths[-1])
                    _chunk_vis_frames = [f for f in _vis_reader]
                    _vis_reader.close()
                    if chunk_idx == 0:
                        full_vis_frames.extend(_chunk_vis_frames)
                    else:
                        _prev_end = starts[chunk_idx - 1] + chunk_size
                        _vis_overlap = max(0, min(_prev_end - s, len(_chunk_vis_frames)))
                        full_vis_frames.extend(_chunk_vis_frames[_vis_overlap:])
                    print(f"[chunk] Collected {len(_chunk_vis_frames)} vis frames from chunk {chunk_idx} "
                          f"(total so far: {len(full_vis_frames)})")
                except Exception as _e:
                    print(f"[chunk] Warning: failed to read vis frames from {chunk_vis_tmp_paths[-1]}: {_e}")

            # Concatenate: drop the first `overlap` frames for every chunk after
            # the first (hard-cut blending). When the tail chunk shifts backward
            # due to remaining-frames handling, compute the actual overlap with
            # the previously appended frames and drop that many.
            if chunk_idx == 0:
                full_video.extend(list(chunk_video))
            else:
                prev_end = starts[chunk_idx - 1] + chunk_size
                actual_overlap = prev_end - s
                actual_overlap = max(0, min(actual_overlap, len(chunk_video)))
                if actual_overlap > 0:
                    full_video.extend(list(chunk_video)[actual_overlap:])
                else:
                    full_video.extend(list(chunk_video))

            # Prepare reference image for the NEXT chunk: take the frame from the
            # current chunk's output that corresponds to the next chunk's start
            # frame in the global timeline. i.e. ref_for_next = chunk_video[next_s - s].
            # This makes the next chunk treat its first frame as its reference image.
            if chunk_idx + 1 < len(starts):
                next_s = starts[chunk_idx + 1]
                local_idx = next_s - s  # index within current chunk_video
                local_idx = max(0, min(local_idx, len(chunk_video) - 1))
                current_ref_image = _frame_to_pil(
                    list(chunk_video)[local_idx], width, height
                )
                print(f"[chunk] Next chunk will use chunk_video[{local_idx}] "
                      f"(= global frame {next_s}) as reference_image")

        video = full_video
        print(f"[chunk] Finished chunked inference: concatenated {len(video)} frames from {len(starts)} chunks")
        # Sanity-check: concatenated length must equal num_frames (the 4k+1-aligned target).
        if len(video) != num_frames:
            print(f"[chunk] WARNING: concatenated length {len(video)} != expected num_frames {num_frames}. "
                  f"Padding/truncating to match.")
            if len(video) > num_frames:
                video = video[:num_frames]
            else:
                # Pad by repeating last frame
                while len(video) < num_frames:
                    video.append(video[-1])

        # Merge per-chunk vis_warp_keypoints videos into the final visualization file.
        if vis_kps_enabled and full_vis_frames:
            try:
                import imageio as _imageio
                os.makedirs(os.path.dirname(final_vis_kps_path) if os.path.dirname(final_vis_kps_path) else ".", exist_ok=True)
                _imageio.mimsave(final_vis_kps_path, full_vis_frames, fps=15)
                print(f"[chunk] Merged {len(full_vis_frames)} vis frames → {final_vis_kps_path}")
            except Exception as _e:
                print(f"[chunk] Warning: failed to write merged vis video: {_e}")
            # Clean up per-chunk temp files
            for _tmp in chunk_vis_tmp_paths:
                try:
                    if os.path.exists(_tmp):
                        os.remove(_tmp)
                except Exception:
                    pass
            # Restore args.vis_warp_keypoints_path to the final merged path so that
            # the downstream create_grid_video call uses the correct (full) file.
            args.vis_warp_keypoints_path = final_vis_kps_path
    else:
        video = _run_pipe_chunk(
            pipe_control_video, key_points, depth_keypoints, depth_keypoints2, num_frames,
            reference_image,
        )
    # Remove padding frames to match original input video length
    if original_num_frames is not None and len(video) > original_num_frames:
        print(f"Removing padding: trimming output from {len(video)} to {original_num_frames} frames")
        video = video[:original_num_frames]

    # Resize back to original resolution if needed
    if original_height is not None and original_width is not None and (height != original_height or width != original_width):
        print(f"Resizing output video from {width}x{height} to original resolution {original_width}x{original_height}")
        resized_video = []
        for frame in video:
            if isinstance(frame, torch.Tensor):
                frame_np = frame.cpu().numpy()
                if frame_np.ndim == 3 and frame_np.shape[0] in [1, 3, 4]:
                    frame_np = np.transpose(frame_np, (1, 2, 0))
                if frame_np.dtype != np.uint8:
                    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                pil_frame = Image.fromarray(frame_np)
            elif isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                pil_frame = Image.fromarray(frame)
            elif isinstance(frame, Image.Image):
                pil_frame = frame
            else:
                pil_frame = frame
            
            if isinstance(pil_frame, Image.Image):
                pil_frame = pil_frame.resize((original_width, original_height), Image.LANCZOS)
            resized_video.append(pil_frame)
        video = resized_video

    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(video, output_path, fps=output_fps, quality=5)
        print(f"Saved video to {output_path} with fps {output_fps}")

        # Create 2x2 grid video when vis_warp_keypoints is enabled
        if getattr(args, 'vis_warp_keypoints', False) and hasattr(args, 'vis_warp_keypoints_path') and args.vis_warp_keypoints_path:
            effective_ctrl_path = control_video_path if control_video_path is not None else control_video_3dpose_path
            grid_stem = os.path.splitext(os.path.basename(output_path))[0]
            grid_dir = os.path.dirname(output_path) or "."
            grid_output_path = os.path.join(grid_dir, f"{grid_stem}_grid.mp4")
            try:
                create_grid_video(
                    control_video_path=effective_ctrl_path,
                    vis_warp_kps_path=args.vis_warp_keypoints_path,
                    reference_image_path=reference_image_path,
                    generated_video_path=output_path,
                    grid_output_path=grid_output_path,
                    fps=output_fps,
                )
            except Exception as e:
                print(f"[create_grid_video] Failed to create grid video: {e}")

        if save_frames:
            # Save each frame as image for debugging
            frames_dir = os.path.splitext(output_path)[0]
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(video):
                if isinstance(frame, Image.Image):
                    frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                elif isinstance(frame, np.ndarray):
                    Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                elif isinstance(frame, torch.Tensor):
                    frame_np = frame.cpu().numpy()
                    if frame_np.ndim == 3 and frame_np.shape[0] in [1, 3, 4]:
                        frame_np = np.transpose(frame_np, (1, 2, 0))
                    if frame_np.dtype != np.uint8:
                        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(frame_np).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            print(f"Saved {len(video)} frames to {frames_dir}")


# Determine if batch mode (reference_image is a directory)
is_batch_mode = os.path.isdir(args.reference_image)

if is_batch_mode:
    # Batch mode: iterate over control video files, match by name with reference images in folder
    ref_dir = args.reference_image
    # Collect reference images indexed by stem (filename without extension)
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    ref_images_by_stem = {}
    for fname in os.listdir(ref_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTS:
            ref_images_by_stem[stem] = os.path.join(ref_dir, fname)
    print(f"Batch mode: found {len(ref_images_by_stem)} reference images in {ref_dir}")

    # Determine control video directory
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    control_dir = None
    control_is_3dpose = False
    if args.control_video is not None:
        control_dir = args.control_video if os.path.isdir(args.control_video) else os.path.dirname(args.control_video)
    elif args.control_video_3dpose is not None:
        control_dir = args.control_video_3dpose if os.path.isdir(args.control_video_3dpose) else os.path.dirname(args.control_video_3dpose)
        control_is_3dpose = True
    assert control_dir is not None, "Either --control_video or --control_video_3dpose must be provided"

    # Determine key_points directory (if provided)
    key_points_dir = None
    if args.key_points is not None:
        key_points_dir = args.key_points if os.path.isdir(args.key_points) else os.path.dirname(args.key_points)

    # Determine depth_keypoints directory (if provided)
    depth_keypoints_dir = None
    if args.depth_keypoints is not None:
        depth_keypoints_dir = args.depth_keypoints if os.path.isdir(args.depth_keypoints) else os.path.dirname(args.depth_keypoints)

    # Determine depth_keypoints2 directory (if provided)
    depth_keypoints2_dir = None
    if args.depth_keypoints2 is not None:
        depth_keypoints2_dir = args.depth_keypoints2 if os.path.isdir(args.depth_keypoints2) else os.path.dirname(args.depth_keypoints2)

    # Iterate over control videos, find matching reference image by stem
    control_files = sorted([f for f in os.listdir(control_dir) if os.path.splitext(f)[1].lower() in VIDEO_EXTS])
    print(f"Batch mode: found {len(control_files)} control videos in {control_dir}")

    # Apply eval_limit to only test the first N videos
    if args.eval_limit is not None and args.eval_limit > 0:
        original_count = len(control_files)
        control_files = control_files[:args.eval_limit]
        print(f"eval_limit={args.eval_limit}: testing first {len(control_files)} of {original_count} videos")

    # Multi-node data parallelism: shard control_files across nodes so that each
    # node independently processes a subset of videos.  Each node internally keeps
    # its own USP (intra-node) parallelism.  Use stride-based slicing (node_rank::num_nodes)
    # so the workload is evenly spread even when len(control_files) is not divisible
    # by num_nodes.
    if args.num_nodes > 1:
        total_count = len(control_files)
        control_files = control_files[args.node_rank::args.num_nodes]
        print(f"Multi-node mode: node_rank={args.node_rank}/{args.num_nodes}, "
              f"this node handles {len(control_files)} of {total_count} videos")

    for ctrl_fname in control_files:
        ctrl_stem = os.path.splitext(ctrl_fname)[0]
        if ctrl_stem not in ref_images_by_stem:
            print(f"Skipping control video {ctrl_fname}: no matching reference image found")
            continue

        ref_img_path = ref_images_by_stem[ctrl_stem]
        ctrl_video_path = os.path.join(control_dir, ctrl_fname)

        # Build paths
        cv_path = None if control_is_3dpose else ctrl_video_path
        cv_3dpose_path = ctrl_video_path if control_is_3dpose else None

        kp_path = None
        if key_points_dir is not None:
            # Try to find matching key_points file (.npz)
            kp_candidate = os.path.join(key_points_dir, ctrl_stem + ".npz")
            if os.path.exists(kp_candidate):
                kp_path = kp_candidate
            else:
                print(f"Warning: key_points file not found for {ctrl_stem}, skipping key_points")

        dk_path = None
        if depth_keypoints_dir is not None:
            # Try to find matching depth_keypoints file (.npz)
            dk_candidate = os.path.join(depth_keypoints_dir, ctrl_stem + ".npz")
            if os.path.exists(dk_candidate):
                dk_path = dk_candidate
            else:
                print(f"Warning: depth_keypoints file not found for {ctrl_stem}, skipping depth_keypoints")

        dk2_path = None
        if depth_keypoints2_dir is not None:
            # Try to find matching depth_keypoints2 file (.npz)
            dk2_candidate = os.path.join(depth_keypoints2_dir, ctrl_stem + ".npz")
            if os.path.exists(dk2_candidate):
                dk2_path = dk2_candidate
            else:
                print(f"Warning: depth_keypoints2 file not found for {ctrl_stem}, skipping depth_keypoints2")

        # Reuse the exp_name logic from args.output or build one
        if args.output is not None:
            # Use user-specified output as directory base
            out_dir = args.output if os.path.isdir(args.output) or not os.path.exists(args.output) else os.path.dirname(args.output)
        else:
            exp_name = "LatentDance"
            out_dir = os.path.join("output", exp_name)

        # Look up per-video prompt from caption CSV if available
        video_prompt = None
        if caption_map is not None:
            video_prompt = caption_map.get(ctrl_stem, None)
            if video_prompt is None:
                print(f"Warning: no caption found for '{ctrl_stem}' in caption CSV, using default prompt")

        # Build output path for this sample
        # Append _csv suffix when using per-video prompt from caption CSV
        # Append _f{N} tag when --num_frames is explicitly specified, so that
        # runs with different frame counts produce distinct output filenames.
        num_frames_tag = f"_f{args.num_frames}" if args.num_frames is not None else ""
        # Append _chunk{size}o{overlap} tag when chunked inference is enabled,
        # so chunked vs non-chunked runs produce distinct output filenames.
        if getattr(args, "chunk_inference_v2", False):
            chunk_tag = f"_chunkv2{args.chunk_size}"
        elif getattr(args, "chunk_inference", False):
            chunk_tag = f"_chunk{args.chunk_size}o{args.chunk_overlap}"
        else:
            chunk_tag = ""
        if video_prompt is not None:
            video_name = f"{ctrl_stem}_csv{num_frames_tag}{chunk_tag}{args.video_suffix}.mp4"
        else:
            video_name = f"{ctrl_stem}{num_frames_tag}{chunk_tag}{args.video_suffix}.mp4"

        output_path = os.path.join(out_dir, args.eval_output_suffix, video_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Batch inference: {ctrl_fname} + {os.path.abspath(ref_img_path)}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        run_single_inference(
            pipe=pipe,
            args=args,
            reference_image_path=os.path.abspath(ref_img_path) if ref_img_path else None,
            control_video_path=os.path.abspath(cv_path) if cv_path else None,
            control_video_3dpose_path=os.path.abspath(cv_3dpose_path) if cv_3dpose_path else None,
            key_points_path=os.path.abspath(kp_path) if kp_path else None,
            output_path=os.path.abspath(output_path) if output_path else None,
            save_frames=False,  # Batch mode: only save video, no frame images
            prompt_override=video_prompt,
            depth_keypoints_path=os.path.abspath(dk_path) if dk_path else None,
            depth_keypoints2_path=os.path.abspath(dk2_path) if dk2_path else None,
        )

else:
    print(f"\n{'='*60}")
    print(f"Single inference: {os.path.abspath(args.reference_image) if args.reference_image else None}")
    print(f"Control video: {os.path.abspath(args.control_video) if args.control_video else None}")
    if args.key_points is not None:
        print(f"Key points: {os.path.abspath(args.key_points)}")
    if args.depth_keypoints is not None:
        print(f"Depth keypoints: {os.path.abspath(args.depth_keypoints)}")
    if args.depth_keypoints2 is not None:
        print(f"Depth keypoints2: {os.path.abspath(args.depth_keypoints2)}")
    print(f"Control video 3dpose: {os.path.abspath(args.control_video_3dpose) if args.control_video_3dpose else None}")
    print(f"Output: {os.path.abspath(args.output) if args.output else None}")
    print(f"{'='*60}")

    # Single mode (original behavior)
    run_single_inference(
        pipe=pipe,
        args=args,
        reference_image_path=os.path.abspath(args.reference_image) if args.reference_image else None,
        control_video_path=os.path.abspath(args.control_video) if args.control_video else None,
        control_video_3dpose_path=os.path.abspath(args.control_video_3dpose) if args.control_video_3dpose else None,
        key_points_path=os.path.abspath(args.key_points) if args.key_points else None,
        output_path=os.path.abspath(args.output) if args.output else None,
        save_frames=True,
        depth_keypoints_path=os.path.abspath(args.depth_keypoints) if args.depth_keypoints else None,
        depth_keypoints2_path=os.path.abspath(args.depth_keypoints2) if args.depth_keypoints2 else None,
    )