import torch, os, argparse, accelerate, warnings
import numpy as np
import random
from collections import defaultdict
from diffsynth.core import UnifiedDataset
from diffsynth.core.loader.file import load_state_dict
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath, DataProcessingOperator
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OrientationGroupedBatchSampler(torch.utils.data.Sampler):
    """
    A batch sampler that groups dataset samples by video orientation (landscape/portrait/square)
    so that within each batch, all samples have the same spatial dimensions after ImageCropAndResize
    with keep_original_ratio=True. This prevents shape mismatches during torch.cat in batched training.
    
    Reads the 'orientation' field directly from metadata CSV (pre-computed by filter_short_videos.py),
    requiring zero additional file I/O at training startup.
    """
    def __init__(self, dataset, batch_size, target_height, target_width, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.drop_last = drop_last
        self.orientation_groups = self._build_orientation_groups()

    def _get_item_metadata(self, idx):
        """
        Get the raw metadata dict for a dataset item without loading the actual data.
        Handles both UnifiedDataset and ConcatDataset (with repeat).
        """
        ds = self.dataset
        if isinstance(ds, torch.utils.data.ConcatDataset):
            # Walk through sub-datasets to find the right one
            remaining = idx
            for sub_ds in ds.datasets:
                if remaining < len(sub_ds):
                    if hasattr(sub_ds, 'data') and sub_ds.data:
                        real_idx = remaining % len(sub_ds.data)
                        return sub_ds.data[real_idx]
                    return None
                remaining -= len(sub_ds)
            return None
        else:
            if hasattr(ds, 'data') and ds.data:
                real_idx = idx % len(ds.data)
                return ds.data[real_idx]
            return None

    def _build_orientation_groups(self):
        """
        Group all dataset sample indices by video orientation.
        Reads the 'orientation' field from metadata (pre-computed in CSV).
        Falls back to 'match' group if the field is missing.
        """
        groups = defaultdict(list)
        total = len(self.dataset)
        target_is_landscape = self.target_width > self.target_height
        missing_count = 0

        print(f"[OrientationGroupedBatchSampler] Grouping {total} samples by orientation from metadata...", flush=True)

        for idx in range(total):
            item_data = self._get_item_metadata(idx)

            if item_data is None:
                groups['match'].append(idx)
                missing_count += 1
                continue

            orientation = item_data.get('orientation', None)
            if orientation is None:
                # No orientation field in metadata — assume matches target
                groups['match'].append(idx)
                missing_count += 1
                continue

            # Determine effective group based on whether video orientation matches target
            video_is_landscape = (orientation == 'landscape')
            if orientation == 'square':
                effective_key = 'match'
            elif video_is_landscape == target_is_landscape:
                effective_key = 'match'
            else:
                effective_key = 'swapped'

            groups[effective_key].append(idx)

        for key, indices in groups.items():
            print(f"[OrientationGroupedBatchSampler] Group '{key}': {len(indices)} samples", flush=True)
        if missing_count > 0:
            print(f"[OrientationGroupedBatchSampler] WARNING: {missing_count} samples missing 'orientation' field in metadata. "
                  f"Re-run filter_short_videos.py to add orientation info to CSV.", flush=True)

        return dict(groups)

    def __iter__(self):
        # Shuffle within each group and produce batches
        all_batches = []
        for key, indices in self.orientation_groups.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            # Chunk into batches
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # Shuffle batch order for randomness across orientations
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for key, indices in self.orientation_groups.items():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


class LoadKeypoints(DataProcessingOperator):
    def __init__(self, num_frames, time_division_factor=1, time_division_remainder=0, height=None, width=None, control_video_path=None):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.height = height
        self.width = width
        self.control_video_path = control_video_path

    def get_num_frames(self, total_frames):
        """Match LoadVideo's temporal processing logic"""
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def get_original_size(self, path):
        # If control_video_path is provided, use it directly (for inference)
        if self.control_video_path is not None:
            video_path = self.control_video_path
            try:
                import imageio
                reader = imageio.get_reader(video_path)
                meta = reader.get_meta_data()
                reader.close()
                return meta['size']  # (width, height)
            except Exception as e:
                print(f"Warning: Failed to read control video {video_path}: {e}")
                return None
        
        # Default logic: try to find the corresponding video file (for training)
        extensions = [".mp4", ".avi", ".mov", ".mkv"]
        # Support multiple keypoints directory patterns:
        #   /key_points/      -> /pose/       (2D DWPose)
        #   /3d_key_points/   -> /3dpose/     (3D SMPL+DWPose)
        #   /pose2_keypoints/ -> /pose2/      (pose2 DWPose from infer_preprocess_yyx)
        #   /pose3_keypoints/ -> /pose3/      (pose3 DWPose with raw score from infer_preprocess_yyx_new)
        if "/3d_key_points/" in path:
            base_path = path.replace("/3d_key_points/", "/3dpose/").rsplit(".", 1)[0]
        elif "/pose3_keypoints/" in path:
            base_path = path.replace("/pose3_keypoints/", "/pose3/").rsplit(".", 1)[0]
        elif "/pose2_keypoints/" in path:
            base_path = path.replace("/pose2_keypoints/", "/pose2/").rsplit(".", 1)[0]
        else:
            base_path = path.replace("/key_points/", "/pose/").rsplit(".", 1)[0]
        
        video_path = None
        for ext in extensions:
            if os.path.exists(base_path + ext):
                video_path = base_path + ext
                break
        
        if video_path is None:
            # Fallback: try to find in the same directory if structure is different
            # This is a best-effort guess
            return None

        try:
            import imageio
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            reader.close()
            return meta['size'] # (width, height)
        except:
            return None

    def parse_frame_data(self, frame_data):
        if isinstance(frame_data, np.ndarray) and frame_data.ndim == 0:
            frame_data = frame_data.item()
        
        candidate = None
        subset = None
        
        if isinstance(frame_data, dict):
            if 'bodies' in frame_data:
                bodies = frame_data['bodies']
                candidate = bodies['candidate']
                subset = bodies['subset']
            elif 'candidate' in frame_data:
                candidate = frame_data['candidate']
                subset = frame_data.get('subset', None)
        else:
            candidate = frame_data
            
        return candidate, subset

    def __call__(self, path):
        try:
            if path.endswith('.npz'):
                data = np.load(path, allow_pickle=True)
                if 'bodies_candidate' in data:
                    bodies_data = data['bodies_candidate']
                else:
                    keys = list(data.keys())
                    bodies_data = data[keys[0]]
            else:
                bodies_data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((self.num_frames, 18, 2))

        total_frames = len(bodies_data)
        # Match LoadVideo's temporal processing logic
        num_frames = self.get_num_frames(total_frames)

        # Detect SMPL 3D keypoints format: float array with shape (T, N, 3)
        # where columns are [x_norm, y_norm, validity].
        # Generated by process_pose_yyx_keypoints_smpl.py, 134 points per frame.
        is_smpl_format = (
            isinstance(bodies_data, np.ndarray)
            and bodies_data.dtype.kind == 'f'  # float dtype
            and bodies_data.ndim == 3
            and bodies_data.shape[2] == 3
        )

        if is_smpl_format:
            # SMPL format: directly slice frames, no need for parse_frame_data
            keypoints = bodies_data[:num_frames].copy()
            # Pad if num_frames > total_frames
            if num_frames > total_frames:
                pad_count = num_frames - total_frames
                last_frame = bodies_data[-1:]
                keypoints = np.concatenate([keypoints, np.repeat(last_frame, pad_count, axis=0)], axis=0)
        else:
            # DWPose format: per-frame parsing with candidate/subset
            selected_frames = []
            for i in range(num_frames):
                idx = i
                if idx >= total_frames:
                    idx = total_frames - 1 
                
                frame_data = bodies_data[idx]
                candidate, subset = self.parse_frame_data(frame_data)
                # print(candidate.shape, subset.shape);assert 0 # (18, 2) (1, 18)
                
                # concat candidate and subset
                if subset is not None:
                    # print(np.max(candidate), np.min(candidate), np.max(subset), np.min(subset), subset, candidate[0]);assert 0 
                    # 1.452224133413337 0.0019566682832581656 17.0 -1.0
                    # 0.9993948051539648 0.268444127344992 17.0 0.0 [[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.]] [0.51766221 0.29122705]
                    candidate = np.concatenate([candidate, subset.T], axis=1)
                    # print(candidate.shape);assert 0 # (18, 3)
                selected_frames.append(candidate)
            keypoints = np.array(selected_frames)

        orig_size = self.get_original_size(path)
        # orig_size = None
        
        # print(orig_size, self.height, self.width);assert 0 # (512, 896) 480 832  

        if orig_size is not None and self.height is not None and self.width is not None:
            orig_w, orig_h = orig_size
            target_h, target_w = self.height, self.width
            
            # Match ImageCropAndResize keep_original_ratio logic:
            # If original image orientation doesn't match target orientation, swap target h/w
            orig_is_landscape = orig_w > orig_h
            target_is_landscape = target_w > target_h
            if orig_is_landscape != target_is_landscape:
                target_h, target_w = target_w, target_h
            
            # Calculate scale (same as ImageCropAndResize.crop_and_resize)
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = round(orig_w * scale)
            new_h = round(orig_h * scale)
            
            # Calculate center crop offset in normalized coordinates
            # keypoints are normalized (0~1.5), representing x/orig_w, y/orig_h
            # After scaling: new image size is (new_w, new_h), then center crop to (target_w, target_h)
            # Crop offset in pixels: dx = (new_w - target_w) / 2, dy = (new_h - target_h) / 2
            # Convert to normalized offset relative to new scaled image size
            dx_normalized = (new_w - target_w) / 2 / new_w
            dy_normalized = (new_h - target_h) / 2 / new_h
            
            # Scale factors from original to scaled image (in normalized space, this is just the ratio)
            # After scaling, normalized coords stay the same (x_pixel * scale / (orig_w * scale) = x_pixel / orig_w)
            # After crop, we need to adjust: new_normalized = (old_normalized - offset) * (new_size / target_size)
            scale_x = new_w / target_w
            scale_y = new_h / target_h
            
            # Apply transformation: subtract crop offset, then scale to new target space
            keypoints[:, :, 0] = (keypoints[:, :, 0] - dx_normalized) * scale_x
            keypoints[:, :, 1] = (keypoints[:, :, 1] - dy_normalized) * scale_y
            
        return torch.tensor(keypoints, dtype=torch.float32)


class LoadDepthKeypoints(DataProcessingOperator):
    """Load depth keypoints from .npz files produced by DVD depth estimation.
    
    The .npz file contains:
        - keypoint_depth: [T, N] float32, globally normalized depth [0, 1] at each keypoint
        - keypoint_depth_valid: [T, N] bool, True if keypoint is valid
        - depth_min: scalar, global min of raw depth
        - depth_max: scalar, global max of raw depth
    
    Returns a tensor of shape [T_selected, N, 2] where:
        - [:, :, 0] = depth value (normalized [0, 1])
        - [:, :, 1] = validity flag (0 or 1)
    """
    def __init__(self, num_frames, time_division_factor=1, time_division_remainder=0):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder

    def get_num_frames(self, total_frames):
        """Match LoadVideo's temporal processing logic"""
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, path):
        try:
            data = np.load(path, allow_pickle=True)
            kp_depth = data['keypoint_depth']       # [T, N] float32
            kp_valid = data['keypoint_depth_valid']  # [T, N] bool
        except Exception as e:
            print(f"Error loading depth keypoints {path}: {e}")
            return torch.zeros((self.num_frames, 128, 2))

        total_frames = kp_depth.shape[0]
        num_frames = self.get_num_frames(total_frames)
        N = kp_depth.shape[1]

        # Select frames (matching LoadVideo temporal logic)
        depth_selected = kp_depth[:num_frames]
        valid_selected = kp_valid[:num_frames].astype(np.float32)

        # Pad if needed
        if num_frames > total_frames:
            pad_count = num_frames - total_frames
            depth_selected = np.concatenate([depth_selected, np.repeat(depth_selected[-1:], pad_count, axis=0)], axis=0)
            valid_selected = np.concatenate([valid_selected, np.repeat(valid_selected[-1:], pad_count, axis=0)], axis=0)

        # Stack depth and validity into [T, N, 2]
        result = np.stack([depth_selected, valid_selected], axis=-1)  # [T, N, 2]
        return torch.tensor(result, dtype=torch.float32)


class LoadDepthKeypoints2(DataProcessingOperator):
    """Load full-frame depth map from .npz files and pre-compute depth_indices in DataLoader.
    
    The .npz file contains:
        - depth: [T, H, W] float16, globally normalized depth [0, 1]
        - depth_min: scalar, global min of raw depth
        - depth_max: scalar, global max of raw depth
    
    Instead of returning the raw [T, H, W] depth map (which can be 200MB+),
    this operator performs temporal downsampling, spatial average pooling, and
    depth quantization directly in the DataLoader worker, returning a compact
    int64 tensor of shape [f, h, w] (~0.5MB) containing depth level indices.
    
    This avoids:
    1. Huge IO overhead from loading/transferring 200MB+ tensors per sample
    2. GPU->CPU synchronization in _build_depth_aware_freqs_v2
    3. Wasted GPU memory and bandwidth for large intermediate tensors
    """
    def __init__(self, num_frames, time_division_factor=1, time_division_remainder=0,
                 depth_levels=64, patch_size_t=1, patch_size_h=8, patch_size_w=8):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.depth_levels = depth_levels
        # Patch sizes for downsampling: pt for temporal, ph/pw for spatial
        # For Wan2.2 14B I2V: patch_size = (1, 2, 2), but the latent is already 1/8 of pixel,
        # so effective pixel patch = (1*4, 2*8, 2*8) = (4, 16, 16)
        # f = num_frames // 4 (from VAE temporal compression)
        # h = H // 16, w = W // 16 (from VAE spatial + patch spatial)
        self.patch_size_t = patch_size_t  # temporal patch size in pixel frames (after VAE: 4)
        self.patch_size_h = patch_size_h  # spatial patch size in pixels (after VAE+patch: 16)
        self.patch_size_w = patch_size_w  # spatial patch size in pixels (after VAE+patch: 16)

    def get_num_frames(self, total_frames):
        """Match LoadVideo's temporal processing logic"""
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, path):
        try:
            data = np.load(path, allow_pickle=True)
            depth_map = data['depth'].astype(np.float32)  # [T, H, W] float16 -> float32
        except Exception as e:
            print(f"Error loading depth map {path}: {e}")
            # Return dummy depth_indices with default mid-level depth
            # Estimate patch grid size: f = num_frames // 4, h = 480 // 16 = 30, w = 832 // 16 = 52
            f_est = max(1, (self.num_frames - 1) // 4 + 1)
            return torch.full((f_est, 30, 52), self.depth_levels // 2, dtype=torch.int64)

        total_frames = depth_map.shape[0]
        num_frames = self.get_num_frames(total_frames)

        # Select frames (matching LoadVideo temporal logic)
        depth_selected = depth_map[:num_frames]

        # Pad if needed
        if num_frames > total_frames:
            pad_count = num_frames - total_frames
            depth_selected = np.concatenate(
                [depth_selected, np.repeat(depth_selected[-1:], pad_count, axis=0)], axis=0
            )

        dm = torch.tensor(depth_selected, dtype=torch.float32)  # [T, H, W]
        T_pixel, H_pixel, W_pixel = dm.shape

        # Temporal downsampling: average every patch_size_t frames
        # For training with num_frames=81, VAE temporal compression factor=4:
        # f = (81 - 1) // 4 + 1 = 21 patch frames
        pt = self.patch_size_t
        f = (T_pixel - 1) // pt + 1 if pt > 1 else T_pixel
        # Pad temporal dimension to be divisible by pt
        T_padded = f * pt
        if T_pixel < T_padded:
            pad_count = T_padded - T_pixel
            dm = torch.cat([dm, dm[-1:].expand(pad_count, -1, -1)], dim=0)
        dm_temporal = dm[:T_padded].reshape(f, pt, H_pixel, W_pixel).mean(dim=1)  # [f, H, W]

        # Spatial downsampling: adaptive average pooling to patch grid resolution
        h = H_pixel // self.patch_size_h
        w = W_pixel // self.patch_size_w
        dm_patch = torch.nn.functional.adaptive_avg_pool2d(
            dm_temporal.unsqueeze(1), (h, w)
        ).squeeze(1)  # [f, h, w]

        # Quantize to integer depth indices [0, depth_levels-1]
        depth_indices = (dm_patch * (self.depth_levels - 1)).long().clamp(0, self.depth_levels - 1)

        return depth_indices  # [f, h, w] int64, ~0.5MB instead of ~200MB


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        args=None,
    ):
        super().__init__()

        self.args = args 

        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        # audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        audio_processor_config = None if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        
        ### init dit.has_ref_conv
        # self.pipe.dit.has_ref_conv = True
        # self.pipe.dit.ref_conv = torch.nn.Conv2d(16, self.pipe.dit.dim, kernel_size=(2, 2), stride=(2, 2))

        # print("Initializing ref_conv layer...")
        # torch.nn.init.zeros_(self.pipe.dit.ref_conv.weight)
        # if self.pipe.dit.ref_conv.bias is not None:
        #     torch.nn.init.zeros_(self.pipe.dit.ref_conv.bias)

        ### change channel concat to temporal concat
        if getattr(args, "temporal_concat", False):
            pose_input_dim = 4 + 16
            self.pipe.dit.patch_embedding_pose = torch.nn.Conv3d(
                pose_input_dim, self.pipe.dit.dim, kernel_size=self.pipe.dit.patch_size, stride=self.pipe.dit.patch_size)
            # parameters zero init 
            torch.nn.init.zeros_(self.pipe.dit.patch_embedding_pose.weight)
            if self.pipe.dit.patch_embedding_pose.bias is not None:
                torch.nn.init.zeros_(self.pipe.dit.patch_embedding_pose.bias)
            print("Initialized temporal concat pipeline successfully: self.pipe.dit.patch_embedding_pose")


        ### init ref_detail_transfer layers: dedicated Q/K/V/O projections with zero init
        ref_detail_transfer_layers = getattr(args, "ref_detail_transfer_layers", None)
        if getattr(args, "first_as_guidance", False):
            num_blocks = len(self.pipe.dit.blocks)
            dim = self.pipe.dit.dim
            device = self.pipe.device
            dtype = self.pipe.torch_dtype
            if ref_detail_transfer_layers is None:
                # Default: apply to last 1/8 of layers (deeper layers carry more detail),
                # consistent with wan_video.py inference default.
                ref_detail_transfer_layers = list(range(num_blocks * 7 // 8, num_blocks))
            elif isinstance(ref_detail_transfer_layers, str):
                ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
            for block_id in ref_detail_transfer_layers:
                block = self.pipe.dit.blocks[block_id]
                for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                    layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                    torch.nn.init.zeros_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                    setattr(block, name, layer)
            print(f"Initialized ref_detail_transfer layers (zero-init) for blocks: {ref_detail_transfer_layers}")


        if getattr(args, "first_as_guidance_middle", False):
            num_blocks = len(self.pipe.dit.blocks)
            dim = self.pipe.dit.dim
            device = self.pipe.device
            dtype = self.pipe.torch_dtype
            if ref_detail_transfer_layers is None:
                # Default: apply to last 1/8 of layers (deeper layers carry more detail),
                # consistent with wan_video.py inference default.
                # ref_detail_transfer_layers = list(range(num_blocks * 7 // 8, num_blocks))
                ref_detail_transfer_layers = [num_blocks//2, num_blocks//2+1]
            elif isinstance(ref_detail_transfer_layers, str):
                ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
            for block_id in ref_detail_transfer_layers:
                block = self.pipe.dit.blocks[block_id]
                for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                    layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                    torch.nn.init.zeros_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                    setattr(block, name, layer)
            print(f"Initialized ref_detail_transfer layers for blocks: {ref_detail_transfer_layers}")


        if getattr(args, "first_as_guidance_adaptive", False):
            num_blocks = len(self.pipe.dit.blocks)
            dim = self.pipe.dit.dim
            device = self.pipe.device
            dtype = self.pipe.torch_dtype
            if ref_detail_transfer_layers is None:
                # Adaptive mode: apply to first, middle, and last blocks
                ref_detail_transfer_layers = [0, num_blocks // 2, num_blocks - 1]
            elif isinstance(ref_detail_transfer_layers, str):
                ref_detail_transfer_layers = [int(x) for x in ref_detail_transfer_layers.split(",")]
            for block_id in ref_detail_transfer_layers:
                block = self.pipe.dit.blocks[block_id]
                for name in ['ref_detail_q', 'ref_detail_k', 'ref_detail_v', 'ref_detail_o']:
                    layer = torch.nn.Linear(dim, dim).to(device=device, dtype=dtype)
                    torch.nn.init.zeros_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                    setattr(block, name, layer)
                # Learnable adaptive weight per block, initialized to 0 (no effect at start)
                adaptive_weight = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
                block.ref_detail_adaptive_weight = adaptive_weight
            print(f"Initialized first_as_guidance_adaptive ref_detail_transfer layers (zero-init + adaptive weight) for blocks: {ref_detail_transfer_layers}")


        if getattr(args, "first_as_guidance_cross_attn", False):
            # Concat self-attention mode: first-frame clean latent tokens are concatenated
            # with all tokens before self-attention, then removed afterwards.
            # No extra learnable parameters needed — the ref tokens participate
            # directly in the existing self-attention, replacing it in-place.
            num_blocks = len(self.pipe.dit.blocks)
            cross_attn_layers = getattr(args, "ref_detail_transfer_layers", None)
            if cross_attn_layers is None:
                cross_attn_layers = list(range(num_blocks * 7 // 8, num_blocks))
            elif isinstance(cross_attn_layers, str):
                cross_attn_layers = [int(x) for x in cross_attn_layers.split(",")]
            self.pipe.dit.cross_attn_guidance_layers = cross_attn_layers
            print(f"Enabled first_as_guidance_cross_attn (concat self-attention, no extra params) for blocks: {cross_attn_layers}")


        if getattr(args, "attention_warp", False):
            dim = self.pipe.dit.dim
            patch_size = self.pipe.dit.patch_size
            device = self.pipe.device
            dtype = self.pipe.torch_dtype

            ### support ip2v pipeline 
            # old_patch_embedding = self.pipe.dit.patch_embedding
            # self.pipe.dit.in_dim = 52
            # self.pipe.dit.patch_embedding = torch.nn.Conv3d(
            #     self.pipe.dit.in_dim, self.pipe.dit.dim, kernel_size=self.pipe.dit.patch_size, stride=self.pipe.dit.patch_size,
            #     bias=old_patch_embedding.bias is not None).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
            # with torch.no_grad():
            #     self.pipe.dit.patch_embedding.weight[:, :old_patch_embedding.in_channels] = old_patch_embedding.weight
            #     if old_patch_embedding.bias is not None:
            #         self.pipe.dit.patch_embedding.bias[:] = old_patch_embedding.bias
            # print("Initialized attention_warp ip2v pipeline successfully: self.pipe.dit.in_dim = 52")

            # Dedicated patchify layers for control latent (16ch) and image latent (16ch)
            self.pipe.dit.attn_warp_control_patchify = torch.nn.Conv3d(
                16, 16, kernel_size=(1,1,1), stride=(1,1,1)
            ).to(device=device, dtype=dtype)

            # Global Q/K/V/O projections for attention warp (zero-init), executed once before DiT blocks
            # for name in ['attn_warp_q', 'attn_warp_k', 'attn_warp_v', 'attn_warp_o']:
            for name in ['attn_warp_q', 'attn_warp_k']:
                # layer = torch.nn.Linear(16, 16).to(device=device, dtype=dtype)
                layer = torch.nn.Linear(16, 16).to(device=device, dtype=dtype)
                setattr(self.pipe.dit, name, layer)

            # RMSNorm for Q and K (matching SelfAttention norm_q/norm_k)
            from diffsynth.models.wan_video_dit import RMSNorm
            self.pipe.dit.attn_warp_norm_q = RMSNorm(16).to(device=device, dtype=dtype)
            self.pipe.dit.attn_warp_norm_k = RMSNorm(16).to(device=device, dtype=dtype)

            print(f"Initialized attention_warp layers: patchify + global Q/K/V/O/norm_q/norm_k")


        if getattr(args, "fix_missing_warp", False):
            # Expand patch_embedding to accept extra 4 channels for keypoint index embedding
            # New in_dim = original in_dim + 4 (keypoint index embedding channels)
            old_patch_embedding = self.pipe.dit.patch_embedding
            old_in_dim = self.pipe.dit.in_dim
            new_in_dim = old_in_dim + 4  # +4 for keypoint index embedding
            self.pipe.dit.in_dim = new_in_dim
            self.pipe.dit.patch_embedding = torch.nn.Conv3d(
                new_in_dim, self.pipe.dit.dim, kernel_size=self.pipe.dit.patch_size, stride=self.pipe.dit.patch_size,
                bias=old_patch_embedding.bias is not None).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
            with torch.no_grad():
                self.pipe.dit.patch_embedding.weight.zero_()
                self.pipe.dit.patch_embedding.weight[:, :old_patch_embedding.in_channels] = old_patch_embedding.weight
                if old_patch_embedding.bias is not None:
                    self.pipe.dit.patch_embedding.bias[:] = old_patch_embedding.bias
            print(f"Initialized fix_missing_warp: expanded patch_embedding in_dim from {old_in_dim} to {new_in_dim} (+4 keypoint index embedding channels)")

        if getattr(args, "fix_missing_warp_v2", False):
            # V2: Same patch_embedding expansion as V1 (+4 channels for keypoint index embedding)
            # Difference from V1: mask keeps original 0/1 logic, missing keypoints filled with learned embedding in latent
            # NOTE: score_filter is now a fully independent flag (see --score_filter); no longer coupled here.
            old_patch_embedding = self.pipe.dit.patch_embedding
            old_in_dim = self.pipe.dit.in_dim
            new_in_dim = old_in_dim + 4  # +4 for keypoint index embedding
            self.pipe.dit.in_dim = new_in_dim
            self.pipe.dit.patch_embedding = torch.nn.Conv3d(
                new_in_dim, self.pipe.dit.dim, kernel_size=self.pipe.dit.patch_size, stride=self.pipe.dit.patch_size,
                bias=old_patch_embedding.bias is not None).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
            with torch.no_grad():
                self.pipe.dit.patch_embedding.weight.zero_()
                self.pipe.dit.patch_embedding.weight[:, :old_patch_embedding.in_channels] = old_patch_embedding.weight
                if old_patch_embedding.bias is not None:
                    self.pipe.dit.patch_embedding.bias[:] = old_patch_embedding.bias
            print(f"Initialized fix_missing_warp_v2: expanded patch_embedding in_dim from {old_in_dim} to {new_in_dim} (+4 keypoint index embedding channels)")

            # Initialize learnable keypoint index embeddings in DiT model
            # Auto-detect num_kps from the first keypoints file in training data
            max_num_kps = self._detect_num_kps_from_dataset(args)
            self.pipe.dit.kp_index_embedding_16ch = torch.nn.Embedding(max_num_kps, 16).to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype)
            self.pipe.dit.kp_index_embedding_4ch = torch.nn.Embedding(max_num_kps, 4).to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype)
            print(f"Initialized fix_missing_warp_v2: kp_index_embedding_16ch ({max_num_kps} -> 16ch) and kp_index_embedding_4ch ({max_num_kps} -> 4ch)")

        if getattr(args, "fix_missing_warp_v3", False):
            # V3: Only kp_index_embedding_16ch needed (fills missing keypoints in latent).
            # Unlike V2, does NOT generate 4ch keypoint index embedding, so NO patch_embedding expansion.
            # NOTE: score_filter is now a fully independent flag (see --score_filter); no longer coupled here.
            max_num_kps = self._detect_num_kps_from_dataset(args)
            self.pipe.dit.kp_index_embedding_16ch = torch.nn.Embedding(max_num_kps, 16).to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype)
            print(f"Initialized fix_missing_warp_v3: kp_index_embedding_16ch ({max_num_kps} -> 16ch), no patch_embedding expansion")

        if getattr(args, "ip2v", False):
            ### support ip2v pipeline 
            old_patch_embedding = self.pipe.dit.patch_embedding
            self.pipe.dit.in_dim = 52
            self.pipe.dit.patch_embedding = torch.nn.Conv3d(
                self.pipe.dit.in_dim, self.pipe.dit.dim, kernel_size=self.pipe.dit.patch_size, stride=self.pipe.dit.patch_size,
                bias=old_patch_embedding.bias is not None).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
            with torch.no_grad():
                self.pipe.dit.patch_embedding.weight.zero_()
                self.pipe.dit.patch_embedding.weight[:, :old_patch_embedding.in_channels] = old_patch_embedding.weight
                if old_patch_embedding.bias is not None:
                    self.pipe.dit.patch_embedding.bias[:] = old_patch_embedding.bias
            print("Initialized ip2v pipeline successfully: self.pipe.dit.in_dim = 52")


        # Initialize depth_embedding for depth-aware additive embedding (like first_as_guidance_middle)
        # Only created when depth_keypoints2 is in extra_inputs, so pretrained checkpoint loads cleanly
        if extra_inputs is not None and "depth_keypoints2" in extra_inputs.split(","):
            dim = self.pipe.dit.dim
            device = self.pipe.device
            dtype = self.pipe.torch_dtype
            depth_levels = self.pipe.dit.depth_levels  # 64
            self.pipe.dit.depth_embedding = torch.nn.Embedding(depth_levels, dim).to(device=device, dtype=dtype)
            # Zero-init so depth embedding has no effect at the start of training
            torch.nn.init.zeros_(self.pipe.dit.depth_embedding.weight)
            print(f"Initialized depth_embedding (zero-init): {depth_levels} levels, dim={dim}")


        # resume checkpoint
        if getattr(args, "resume_checkpoint_path", None) is not None:
            print(f"Resuming checkpoint from {args.resume_checkpoint_path}")
            state_dict = load_state_dict(args.resume_checkpoint_path)
            self.pipe.dit.load_state_dict(state_dict)
            print(f"Loaded high/low noise SFT model from {args.resume_checkpoint_path}")


        # TOKEN_REPLACE: create a trainable clone of `patch_embedding` named
        # `patch_embedding_token_replace`, weight-initialized from the frozen original so
        # that the first forward pass is numerically identical. During training only this
        # clone is updated; the rest of DiT (including the original `patch_embedding`)
        # stays frozen (handled after `switch_pipe_to_training_mode` below).
        if getattr(args, "token_replace", False):
            _orig_pe = self.pipe.dit.patch_embedding
            _pe_dtype = _orig_pe.weight.dtype
            _pe_device = _orig_pe.weight.device
            _clone_pe = torch.nn.Conv3d(
                _orig_pe.in_channels, _orig_pe.out_channels,
                kernel_size=_orig_pe.kernel_size, stride=_orig_pe.stride,
                bias=_orig_pe.bias is not None,
            ).to(device=_pe_device, dtype=_pe_dtype)
            with torch.no_grad():
                _clone_pe.weight.copy_(_orig_pe.weight)
                if _orig_pe.bias is not None:
                    _clone_pe.bias.copy_(_orig_pe.bias)
            self.pipe.dit.patch_embedding_token_replace = _clone_pe
            print(
                f"[token_replace] Initialized patch_embedding_token_replace "
                f"(in={_orig_pe.in_channels}, out={_orig_pe.out_channels}, "
                f"kernel={_orig_pe.kernel_size}) as a copy of patch_embedding."
            )


        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    @staticmethod
    def _detect_num_kps_from_dataset(args):
        """Auto-detect the number of keypoints from the first keypoints file in training data.
        
        Reads the first metadata CSV, finds the first key_points2 entry,
        loads the .npz file, and returns shape[1] (number of keypoints per frame).
        """
        import csv
        dataset_base_paths = args.dataset_base_path.split(",")
        dataset_metadata_paths = args.dataset_metadata_path.split(",")
        
        for base_path, metadata_path in zip(dataset_base_paths, dataset_metadata_paths):
            try:
                with open(metadata_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        kp_path = row.get('key_points3', None) or row.get('key_points2', None)
                        if kp_path is None or kp_path.strip() == '':
                            continue
                        full_path = os.path.join(base_path, kp_path.strip())
                        if not os.path.exists(full_path):
                            continue
                        data = np.load(full_path, allow_pickle=True)
                        if 'bodies_candidate' in data:
                            bodies_data = data['bodies_candidate']
                        else:
                            keys = list(data.keys())
                            bodies_data = data[keys[0]]
                        num_kps = bodies_data.shape[1]
                        print(f"Auto-detected num_kps={num_kps} from {full_path}")
                        return num_kps
            except Exception as e:
                print(f"Warning: Failed to detect num_kps from {metadata_path}: {e}")
                continue
        
        print("Warning: Could not auto-detect num_kps from training data, falling back to 128")
        return 128

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "input_image_frames":
                # TOKEN_REPLACE: provide the first 5 pixel frames of the training video as
                # the multi-frame image guidance. VAEImageEmbedder picks this up when
                # token_replace=True and encodes them in place of the single-frame input.
                num_token_replace_frames = 5
                inputs_shared["input_image_frames"] = data["video"][:num_token_replace_frames]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "3d_control_video":
                # Map 3D pose control video to the standard control_video key
                # expected by the downstream pipeline
                inputs_shared["control_video"] = data["3d_control_video"]
            elif extra_input == "3d_key_points":
                # Map 3D keypoints (SMPL+DWPose, 134pts) to the standard key_points key
                # expected by WanVideoUnit_DirectWarp pipeline
                inputs_shared["key_points"] = data["3d_key_points"]
            elif extra_input == "control_video2":
                # Map pose2 control video to the standard control_video key
                # expected by the downstream pipeline
                inputs_shared["control_video"] = data["control_video2"]
            elif extra_input == "control_video3":
                # Map pose3 control video to the standard control_video key
                # expected by the downstream pipeline
                inputs_shared["control_video"] = data["control_video3"]
            elif extra_input == "key_points2":
                # Map pose2 keypoints (DWPose 128pts) to the standard key_points key
                # expected by WanVideoUnit_DirectWarp pipeline
                inputs_shared["key_points"] = data["key_points2"]
            elif extra_input == "key_points3":
                # Map pose3 keypoints (DWPose 128pts with raw score) to the standard key_points key
                # Used by both plain warp (LOAD_POSE3_KEY_POINTS) and score-filtered warp (LOAD_POSE3_KEY_POINTS_FIX_MISSING_V2)
                inputs_shared["key_points"] = data["key_points3"]
            elif extra_input == "depth_keypoints":
                # Pass depth keypoints (per-keypoint depth values from DVD depth estimation)
                # to the pipeline for depth-aware RoPE
                inputs_shared["depth_keypoints"] = data["depth_keypoints"]
            elif extra_input == "depth_keypoints2":
                # Pass full-frame depth map [T, H, W] from DVD depth estimation v2
                # for depth-aware RoPE (no sparse interpolation needed)
                inputs_shared["depth_keypoints2"] = data["depth_keypoints2"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        # When force_empty_prompt is enabled, override the CSV prompt with an empty string
        prompt_text = data["prompt"]
        # print(f"prompt_text={prompt_text}", self.args.force_empty_prompt);assert 0
        if getattr(self.args, "force_empty_prompt", False):
            prompt_text = ""
        inputs_posi = {"prompt": prompt_text}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        ### Auto-forward all pipeline_kwargs from args to inputs_shared
        for key in getattr(self.args, "pipeline_kwargs_keys", []):
            inputs_shared[key] = getattr(self.args, key)

        # TOKEN_REPLACE per-step stochastic gating:
        #   When --token_replace is enabled (args.token_replace=True), independently flip a
        #   fair coin for each training forward pass. With probability 0.5 we actually apply
        #   token_replace (5-frame VAE guidance + DiT routed through patch_embedding_token_replace);
        #   with probability 0.5 we fall back to the vanilla single-frame path. When
        #   --token_replace is disabled this branch is a no-op and the feature stays off.
        #
        # Note: under `true_batch` mode `_stack_batch_inputs` preserves the first sample's
        # bool value for the whole batch (tensor stacking is skipped for non-tensor values),
        # and under `micro_batch` each sample forwards independently, so in both modes the
        # effective probability per optimizer step is 0.5.
        if getattr(self.args, "token_replace", False):
            import random as _random
            inputs_shared["token_replace"] = _random.random() < 0.5

        # Forward sanity_check flag to pipeline for visualization in units like DirectWarpFixMissing
        if getattr(self.args, "sanity_check", False):
            inputs_shared["sanity_check"] = True
            if "_sanity_check_data_id" in data:
                inputs_shared["sanity_check_data_id"] = data["_sanity_check_data_id"]
            
        return inputs_shared, inputs_posi, inputs_nega
    
    def _stack_batch_inputs(self, batch_inputs_list):
        """
        Stack a list of (inputs_shared, inputs_posi, inputs_nega) into batched versions.
        Tensors with matching shapes are concatenated along the batch dimension (dim=0).
        Non-tensor values are taken from the first sample (assumed identical across batch).
        """
        batch_shared, batch_posi, batch_nega = {}, {}, {}
        
        for key in batch_inputs_list[0][0]:  # inputs_shared keys
            values = [item[0][key] for item in batch_inputs_list]
            if isinstance(values[0], torch.Tensor):
                batch_shared[key] = torch.cat(values, dim=0)
            else:
                batch_shared[key] = values[0]  # scalars / strings / booleans are same across batch
        
        for key in batch_inputs_list[0][1]:  # inputs_posi keys
            values = [item[1][key] for item in batch_inputs_list]
            if isinstance(values[0], torch.Tensor):
                batch_posi[key] = torch.cat(values, dim=0)
            else:
                batch_posi[key] = values[0]
        
        for key in batch_inputs_list[0][2]:  # inputs_nega keys
            values = [item[2][key] for item in batch_inputs_list]
            if isinstance(values[0], torch.Tensor):
                batch_nega[key] = torch.cat(values, dim=0)
            else:
                batch_nega[key] = values[0]
        
        return batch_shared, batch_posi, batch_nega

    def forward(self, data, inputs=None):
        train_batch_size = getattr(self.args, "train_batch_size", 1) if self.args is not None else 1
        # BATCH_MODE env var: "micro_batch" for per-sample forward + averaged loss + single backward,
        #                     "true_batch" for stacking intermediates + single batched loss.
        # Default is "true_batch" for backward compatibility.
        batch_mode = os.environ.get("BATCH_MODE", "true_batch").lower()

        if train_batch_size > 1 and isinstance(data, list):
            if batch_mode == "micro_batch":
                # Micro-batch accumulation: independent forward per sample, average losses, single backward
                total_loss = 0.0
                num_samples = len(data)
                for sample in data:
                    sample_inputs = self.get_pipeline_inputs(sample)
                    sample_inputs = self.transfer_data_to_device(sample_inputs, self.pipe.device, self.pipe.torch_dtype)
                    for unit in self.pipe.units:
                        sample_inputs = self.pipe.unit_runner(unit, self.pipe, *sample_inputs)
                    sample_loss = self.task_to_loss[self.task](self.pipe, *sample_inputs)
                    total_loss = total_loss + sample_loss
                loss = total_loss / num_samples
                return loss
            else:
                # True batched training: run pipeline units per sample, then stack and compute loss once
                batch_inputs_list = []
                for sample in data:
                    sample_inputs = self.get_pipeline_inputs(sample)
                    sample_inputs = self.transfer_data_to_device(sample_inputs, self.pipe.device, self.pipe.torch_dtype)
                    for unit in self.pipe.units:
                        sample_inputs = self.pipe.unit_runner(unit, self.pipe, *sample_inputs)
                    batch_inputs_list.append(sample_inputs)
                
                # Stack all intermediate results along batch dimension
                batched_shared, batched_posi, batched_nega = self._stack_batch_inputs(batch_inputs_list)
                loss = self.task_to_loss[self.task](self.pipe, batched_shared, batched_posi, batched_nega)
                return loss
        else:
            # Single sample path (original behavior)
            if inputs is None: inputs = self.get_pipeline_inputs(data)
            inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            loss = self.task_to_loss[self.task](self.pipe, *inputs)
            return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="Path to resume checkpoint.")
    parser.add_argument("--sanity_check", default=False, action="store_true", help="Whether to perform a sanity check on the dataset.")
    parser.add_argument("--force_empty_prompt", default=False, action="store_true", help="Force prompt to empty string during training (ignore CSV prompt field).")
    parser.add_argument("--ip2v", default=False, action="store_true", help="Whether to use ip2v.")

    # Pipeline kwargs group: all args in this group will be auto-forwarded to inputs_shared
    # To add a new pipeline kwarg, simply add it here — no changes needed in get_pipeline_inputs().
    pipe_kwargs = parser.add_argument_group("pipeline_kwargs", "Args auto-forwarded to pipeline inputs_shared")
    pipe_kwargs.add_argument("--temporal_concat", default=False, action="store_true", help="Whether to use temporal concat instead of channel concat.")
    pipe_kwargs.add_argument("--first_as_guidance", default=False, action="store_true", help="Whether to use first frame as guidance.")
    pipe_kwargs.add_argument("--first_as_guidance_middle", default=False, action="store_true", help="Whether to use first frame as guidance in the middle of the video.")
    pipe_kwargs.add_argument("--first_as_guidance_adaptive", default=False, action="store_true", help="Whether to use first frame as guidance with adaptive weights on first/middle/last blocks.")
    pipe_kwargs.add_argument("--first_as_guidance_cross_attn", default=False, action="store_true", help="Whether to use first frame as guidance via concat self-attention: concat first-frame clean latent tokens with all tokens for self-attention, then remove them afterwards.")
    pipe_kwargs.add_argument("--attention_warp", default=False, action="store_true", help="Whether to use attention-based warp.")
    pipe_kwargs.add_argument("--fix_missing_warp", default=False, action="store_true", help="Whether to use fix-missing warp (V1): modify mask for missing keypoints and add keypoint index embedding.")
    pipe_kwargs.add_argument("--fix_missing_warp_v2", default=False, action="store_true", help="Whether to use fix-missing warp V2: keep mask 0/1 logic, fill missing keypoints with 16ch learned embedding in latent, +4ch keypoint index embedding appended.")
    pipe_kwargs.add_argument("--fix_missing_warp_v3", default=False, action="store_true", help="Whether to use fix-missing warp V3: same as V2 (mask 0/1, missing filled with 16ch learned embedding) but WITHOUT 4ch keypoint index embedding. No patch_embedding expansion needed.")
    pipe_kwargs.add_argument("--score_filter", default=False, action="store_true", help="Whether to apply score-based filtering on keypoints before warp (body>=0.3, hand>=0.3, face>=0.3, skip face jaw/nose bridge). Fully independent of --fix_missing_warp / --fix_missing_warp_v2 / --fix_missing_warp_v3 (can be combined freely with any of them, or used alone with plain DirectWarp).")
    pipe_kwargs.add_argument("--warp_limbs", default=False, action="store_true", help="Whether to also warp along pose limb connections (body/hand skeleton lines) in addition to keypoint positions.")
    pipe_kwargs.add_argument("--face_skip", default=False, action="store_true", help="Whether to skip drawing/warping FACE_SKIP_IDX keypoints (jaw contour 0-16 and nose bridge 27-35) to match pose visualization behavior. When enabled, these face keypoints are zeroed out inside both score-based filtering and the warp pipeline. Default False; the Shell launcher sets its default to follow WARP_LIMBS to preserve legacy behavior.")
    pipe_kwargs.add_argument("--token_replace", default=False, action="store_true", help="TOKEN_REPLACE: when enabled, feed the first 5 pixel frames of the training video (instead of only the first frame) into the VAE-based image embedder (WanVideoUnit_ImageEmbedderVAE). The per-pixel-frame mask is set to 1 for the first 5 frames and 0 elsewhere, which after Wan VAE temporal compression (factor 4, first frame standalone) yields mask=1 on exactly the first 2 latent frames and mask=0 afterwards. The remaining pixel frames are padded with zeros just like the original single-frame scheme. DiT.patchify then routes through a trainable clone `patch_embedding_token_replace` initialized from the frozen pretrained `patch_embedding`; during training only this clone is updated and the rest of DiT (including the original patch_embedding) stays frozen. Purpose: expose the model to the setting where only the first 5 frames act as temporal guidance, while preserving the pretrained projection.")
    return parser, pipe_kwargs


if __name__ == "__main__":
    parser, pipe_kwargs = wan_parser()
    args = parser.parse_args()

    # Record which keys belong to pipeline_kwargs group for auto-forwarding
    args.pipeline_kwargs_keys = [action.dest for action in pipe_kwargs._group_actions]

    if args.resume_checkpoint_path is None and os.path.exists(args.output_path):
        import re
        checkpoints = []
        for file in os.listdir(args.output_path):
            match = re.search(r'epoch-(\d+)\.safetensors', file)
            if match:
                epoch = int(match.group(1))
                checkpoints.append((epoch, os.path.join(args.output_path, file)))
        
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            args.resume_checkpoint_path = checkpoints[0][1]
            print(f"Auto-resuming from {args.resume_checkpoint_path}")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset_base_paths = args.dataset_base_path.split(",")
    dataset_metadata_paths = args.dataset_metadata_path.split(",")
    if len(dataset_base_paths) != len(dataset_metadata_paths):
        raise ValueError("The number of dataset base paths and metadata paths must be the same.")
    
    datasets = []
    for base_path, metadata_path in zip(dataset_base_paths, dataset_metadata_paths):
        dataset = UnifiedDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator_keep_original_ratio(
                base_path=base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(base_path) >> LoadAudio(sr=16000),
                "key_points": ToAbsolutePath(base_path) >> LoadKeypoints(args.num_frames, 4, 1, height=args.height, width=args.width),
                "3d_key_points": ToAbsolutePath(base_path) >> LoadKeypoints(args.num_frames, 4, 1, height=args.height, width=args.width),
                "key_points2": ToAbsolutePath(base_path) >> LoadKeypoints(args.num_frames, 4, 1, height=args.height, width=args.width),
                "key_points3": ToAbsolutePath(base_path) >> LoadKeypoints(args.num_frames, 4, 1, height=args.height, width=args.width),
                "depth_keypoints": ToAbsolutePath(base_path) >> LoadDepthKeypoints(args.num_frames, 4, 1),
                "depth_keypoints2": ToAbsolutePath(base_path) >> LoadDepthKeypoints2(args.num_frames, 4, 1, patch_size_t=4, patch_size_h=16, patch_size_w=16),
            },
            sanity_check=args.sanity_check,
        )
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    # Create orientation-grouped batch sampler when batch_size > 1 and true_batch mode.
    # In micro_batch mode, samples are forwarded independently so shape mismatch is not an issue;
    # a plain shuffled DataLoader with batch_size is sufficient.
    train_batch_size = getattr(args, "train_batch_size", 1)
    batch_mode = os.environ.get("BATCH_MODE", "true_batch").lower()
    if train_batch_size > 1 and batch_mode != "micro_batch":
        if accelerator.is_main_process:
            print(f"Creating OrientationGroupedBatchSampler (batch_size={train_batch_size}, mode={batch_mode})...", flush=True)
        args._batch_sampler = OrientationGroupedBatchSampler(
            dataset=dataset,
            batch_size=train_batch_size,
            target_height=args.height,
            target_width=args.width,
            drop_last=False,
        )
    else:
        args._batch_sampler = None
        if train_batch_size > 1 and accelerator.is_main_process:
            print(f"Micro-batch mode: skipping OrientationGroupedBatchSampler (batch_size={train_batch_size})", flush=True)

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        args=args,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
