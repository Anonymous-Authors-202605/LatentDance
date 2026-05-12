import torch, types
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d, flash_attention
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunControl_temporal_concat(),
            WanVideoUnit_DirectWarp(),
            WanVideoUnit_DirectWarpFixMissing(),
            WanVideoUnit_DirectWarpFixMissingV2(),
            WanVideoUnit_DirectWarpFixMissingV3(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    def _save_first_frame_attn_heatmaps(self, rec, save_dir, timestep, tag=""):
        """Render per-block, per-head-averaged attention heatmaps showing how
        other frames influence the first-frame latent via self-attention.

        Inputs:
            rec: global FIRST_FRAME_ATTN_REC dict; rec["weights"] is
                 {block_idx: tensor[H_heads, f*h*w] float32 (CPU)}.
            save_dir: output root directory.
            timestep: current diffusion timestep tensor / scalar.
            tag: optional string tag appended to filenames.

        Saves under save_dir/:
            t{T}_block{B}_per_frame_summary.png  : bar chart over T frames
                                                   (mean weight per frame)
            t{T}_block{B}_grid.png               : T small (H,W) heatmaps
                                                   arranged in a grid
            t{T}_block{B}_weights.npz            : raw tensor [H_heads, T, H, W]
        """
        import os
        import numpy as _np

        weights_map = rec.get("weights", {})
        f_val = rec.get("f")
        h_val = rec.get("h")
        w_val = rec.get("w")
        if not weights_map or f_val is None or h_val is None or w_val is None:
            # Nothing recorded (e.g. block loop didn't run, or f/h/w not set).
            return

        # Timestep scalar
        if hasattr(timestep, "item"):
            t_val = timestep.item() if timestep.ndim == 0 else float(timestep[0].item())
        else:
            t_val = float(timestep)

        # Try matplotlib; fall back to cv2-based heatmap if unavailable.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _have_mpl = True
        except Exception:
            _have_mpl = False

        suffix = f"_{tag}" if tag else ""

        for block_idx in sorted(weights_map.keys()):
            w = weights_map[block_idx]  # [H_heads, f*h*w] float32 CPU
            H_heads = w.shape[0]
            try:
                w = w.view(H_heads, f_val, h_val, w_val)  # [H_heads, T, H, W]
            except Exception:
                continue

            w_np = w.numpy()  # [H_heads, T, H, W]
            head_mean = w_np.mean(axis=0)  # [T, H, W]

            # ---- Per-frame summary (bar chart of total weight attracted from
            #      first-frame queries to each frame, i.e. sum over spatial dim)
            per_frame_sum = head_mean.sum(axis=(1, 2))  # [T]

            base_name = f"t{int(t_val)}_block{block_idx:02d}{suffix}"
            npz_path = os.path.join(save_dir, base_name + "_weights.npz")
            _np.savez_compressed(npz_path, head_mean=head_mean, per_frame_sum=per_frame_sum)

            if _have_mpl:
                # Summary bar chart over frames
                fig, ax = plt.subplots(figsize=(max(4, f_val * 0.25), 3))
                ax.bar(_np.arange(f_val), per_frame_sum, color="tab:blue")
                ax.set_xlabel("Frame index (0 = first frame)")
                ax.set_ylabel("Total attention weight from first-frame queries")
                ax.set_title(f"t={int(t_val)}, block={block_idx}: first-frame attention per frame")
                ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, base_name + "_per_frame_summary.png"), dpi=120)
                plt.close(fig)

                # Grid of per-frame spatial heatmaps (head-mean)
                cols = min(8, f_val)
                rows = (f_val + cols - 1) // cols
                fig2, axs = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), squeeze=False)
                vmax = float(head_mean.max()) if head_mean.size else 1.0
                vmin = float(head_mean.min()) if head_mean.size else 0.0
                for i in range(rows * cols):
                    r, c = divmod(i, cols)
                    ax = axs[r][c]
                    if i < f_val:
                        im = ax.imshow(head_mean[i], cmap="hot", vmin=vmin, vmax=vmax)
                        title_color = "red" if i == 0 else "black"
                        ax.set_title(f"f={i}", fontsize=7, color=title_color)
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig2.suptitle(f"t={int(t_val)}, block={block_idx}: attn weight map "
                              f"(first-frame Q -> each frame), head-mean",
                              fontsize=9)
                fig2.tight_layout(rect=[0, 0, 1, 0.96])
                fig2.savefig(os.path.join(save_dir, base_name + "_grid.png"), dpi=120)
                plt.close(fig2)
            else:
                # cv2 fallback: save only the summary as a simple bar image
                import cv2
                # Per-frame bar via numpy
                bar_h = 120
                bar_w = max(200, f_val * 6)
                img = _np.zeros((bar_h, bar_w, 3), dtype=_np.uint8)
                if per_frame_sum.max() > 0:
                    norm = per_frame_sum / per_frame_sum.max()
                else:
                    norm = per_frame_sum
                for i in range(f_val):
                    x0 = int(i * bar_w / max(f_val, 1))
                    x1 = int((i + 1) * bar_w / max(f_val, 1))
                    bh = int(norm[i] * (bar_h - 10))
                    color = (0, 0, 255) if i == 0 else (180, 180, 0)
                    cv2.rectangle(img, (x0, bar_h - bh), (x1 - 1, bar_h - 1), color, -1)
                cv2.imwrite(os.path.join(save_dir, base_name + "_per_frame_summary.png"), img)

        print(f"[first_frame_attn_vis] Saved heatmaps for timestep={int(t_val)}, "
              f"blocks={sorted(weights_map.keys())} -> {save_dir}")


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name
            if dist.is_available() and dist.is_initialized():
                device = get_device_name()
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # TOKEN_REPLACE (inference side): optional list of PIL frames fed to
        # WanVideoUnit_ImageEmbedderVAE in place of the single `input_image` when
        # the `token_replace` pipeline_kwarg is True. Used by chunked inference V2
        # to hand over the previous chunk's last 5 frames as temporal guidance.
        input_image_frames: Optional[list] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # image-pose-to-video, latent warp keypoints
        key_points: Union[torch.Tensor, None] = None,
        # Depth keypoints for depth-aware RoPE
        depth_keypoints: Union[torch.Tensor, None] = None,
        # Camera control        # Full depth map for depth-aware RoPE v2 (dense, [T, H, W])
        depth_keypoints2: Union[torch.Tensor, None] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # # Reference frame detail transfer
        # ref_detail_transfer_scale: Optional[float] = 1.0,
        # ref_detail_transfer_start: Optional[float] = 0.5,
        # ref_detail_transfer_end: Optional[float] = 0.0,
        # ref_detail_transfer_layers: Optional[list] = None,
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
        args: Optional[dict] = None,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_image_frames": input_image_frames,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "key_points": key_points,
            "depth_keypoints": depth_keypoints,
            "depth_keypoints2": depth_keypoints2,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            # "ref_detail_transfer_scale": ref_detail_transfer_scale,
            # "ref_detail_transfer_start": ref_detail_transfer_start,
            # "ref_detail_transfer_end": ref_detail_transfer_end,
            # "ref_detail_transfer_layers": ref_detail_transfer_layers,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
        }

        ### Auto-forward all pipeline_kwargs from args to inputs_shared
        for key in getattr(args, "pipeline_kwargs_keys", []):
            inputs_shared[key] = getattr(args, key)

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        # First-frame self-attention visualization setup
        # -------------------------------------------------
        # When the `first_frame_attn_vis` pipeline_kwarg is enabled, we turn on
        # a module-level recorder inside wan_video_dit.SelfAttention.forward.
        # The recorder stores softmax(Q @ K^T / sqrt(d)) weights from first-frame
        # query tokens to all key tokens. After each (positive) model_fn call we
        # read the recorder, render per-block attention heatmaps, and save them.
        _ffav_enabled = bool(inputs_shared.get("first_frame_attn_vis", False))
        if _ffav_enabled:
            from ..models import wan_video_dit as _wvd
            # Check USP: recorder path only works without sequence-parallel sharding
            try:
                import torch.distributed as _dist
                _is_usp = _dist.is_initialized() and _dist.get_world_size() > 1
            except Exception:
                _is_usp = False
            if _is_usp:
                print("[first_frame_attn_vis] WARNING: USP/sequence parallel is enabled; "
                      "first-frame attention visualization is not supported under USP. "
                      "Run with NPROC_PER_NODE=1 (USE_USP off) to enable.")
                _ffav_enabled = False
            else:
                _ffav_layers_str = inputs_shared.get("first_frame_attn_vis_layers", "") or ""
                _ffav_layers = None
                if _ffav_layers_str:
                    try:
                        _ffav_layers = set(int(x) for x in str(_ffav_layers_str).split(",") if x.strip())
                    except Exception:
                        _ffav_layers = None
                if _ffav_layers is None:
                    # Default: record all blocks (user can reduce via layers).
                    _ffav_layers = set(range(len(self.dit.blocks)))
                _ffav_dir = inputs_shared.get("first_frame_attn_vis_dir", "./output/first_frame_attn_vis")
                os.makedirs(_ffav_dir, exist_ok=True)
                # Video tag for file naming (reuse video_suffix if present)
                _ffav_tag = inputs_shared.get("video_suffix", "") or ""
                print(f"[first_frame_attn_vis] Enabled: layers={sorted(_ffav_layers)}, dir={_ffav_dir}")

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # --- Enable recorder only for the positive model_fn call ---
            if _ffav_enabled:
                from ..models import wan_video_dit as _wvd
                _wvd.FIRST_FRAME_ATTN_REC["enabled"] = True
                _wvd.FIRST_FRAME_ATTN_REC["layers"] = _ffav_layers
                _wvd.FIRST_FRAME_ATTN_REC["weights"] = {}
                # f/h/w will be populated by model_fn_wan_video before the block loop.
                _wvd.FIRST_FRAME_ATTN_REC["f"] = None
                _wvd.FIRST_FRAME_ATTN_REC["h"] = None
                _wvd.FIRST_FRAME_ATTN_REC["w"] = None

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)

            # --- Save recorded attention heatmaps (positive only) ---
            if _ffav_enabled:
                from ..models import wan_video_dit as _wvd
                _wvd.FIRST_FRAME_ATTN_REC["enabled"] = False
                try:
                    self._save_first_frame_attn_heatmaps(
                        rec=_wvd.FIRST_FRAME_ATTN_REC,
                        save_dir=_ffav_dir,
                        timestep=timestep,
                        tag=_ffav_tag,
                    )
                except Exception as _e:
                    print(f"[first_frame_attn_vis] Failed to save heatmaps: {_e}")

            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "input_image_frames", "token_replace"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride, input_image_frames=None, token_replace=False):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        # TOKEN_REPLACE branch: feed the FIRST 5 pixel frames (instead of only the first one)
        # into the VAE-based image embedder. The per-pixel-frame mask is set to 1 for the
        # first 5 frames (which, after Wan VAE temporal compression with factor 4 and the
        # standalone first frame, covers exactly the first 2 latent frames) and 0 elsewhere.
        # The remaining num_frames-5 pixel frames are padded with zeros, matching the
        # original "padding 0 for non-first frames" scheme but extended from 1 -> 5.
        #
        # Notes on the mask reshape that follows (kept identical to the original logic):
        #   msk[:, 0:1] is repeated 4x in time, then concatenated with msk[:, 1:], so the
        #   final time length becomes num_frames + 3, which is reshaped to
        #   [4, (num_frames + 3) // 4, H/8, W/8] == [4, num_latent, H/8, W/8].
        #   With the first 5 pixel-frame mask slots set to 1, the first 2 latent-frame
        #   slots end up all-ones, matching exactly "first 2 VAE latent frames act as
        #   temporal guidance".
        if token_replace and input_image_frames is not None and len(input_image_frames) > 0:
            # Clamp to num_frames so that when the clip is shorter than the nominal 5
            # TOKEN_REPLACE guidance frames (e.g. num_frames=1/2/3/4) we don't produce a
            # negative pad_len. All downstream mask / concat math then stays consistent.
            token_replace_pixel_frames = min(5, int(num_frames))
            frames = list(input_image_frames)[:token_replace_pixel_frames]
            # Defensive: if for some reason fewer frames were supplied, right-pad by repeating
            # the last available frame so the tensor has exactly token_replace_pixel_frames
            # frames. This keeps the mask semantics consistent.
            while len(frames) < token_replace_pixel_frames:
                frames.append(frames[-1])
            processed_frames = []
            for frame in frames:
                processed_frame = pipe.preprocess_image(frame.resize((width, height))).to(pipe.device)  # [1, 3, H, W]
                processed_frames.append(processed_frame)
            # Stack to [3, token_replace_pixel_frames, H, W] (time dim after channel dim,
            # matching what `image.transpose(0, 1)` produces for the single-frame path).
            image_stack = torch.stack([f[0] for f in processed_frames], dim=1)  # [3, <=5, H, W]

            msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
            # Only zero out the "future" slots if there are any; when num_frames <= 5 this
            # slice is empty and the whole mask stays 1, which is exactly what we want.
            if token_replace_pixel_frames < num_frames:
                msk[:, token_replace_pixel_frames:] = 0

            pad_len = num_frames - token_replace_pixel_frames  # guaranteed >= 0
            if pad_len > 0:
                vae_input = torch.concat(
                    [image_stack, torch.zeros(3, pad_len, height, width).to(image_stack.device)],
                    dim=1,
                )
            else:
                vae_input = image_stack

            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]

            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            # print('1', y.shape, msk.shape) # 1 torch.Size([1, 20, 21, 104, 60]) torch.Size([4, 21, 104, 60])
            return {"y": y}

        # Original single-frame path (unchanged behavior).
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # print('1', y.shape, msk.shape) # 1 torch.Size([1, 20, 21, 60, 104]) torch.Size([4, 21, 60, 104])
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None and not (y is None):
            # ip2v pipeline
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            # print(y.shape, y_dim, control_latents.shape, latents.shape);assert 0
            # torch.Size([1, 20, 21, 60, 104]) 20 torch.Size([1, 16, 21, 60, 104]) torch.Size([1, 16, 21, 60, 104])
            y = torch.concat([y, control_latents], dim=1)
            return {"clip_feature": clip_feature, "y": y}

        elif clip_feature is None or y is None:
            # print(clip_feature is None, y is None);assert 0 # True False
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]

        # print(pipe.dit.in_dim, y_dim, control_latents.shape, latents.shape, y.shape)
        # Wan2.2 Fun Control: 52, 20 torch.Size([1, 16, 13, 104, 72]) torch.Size([1, 16, 13, 104, 72]) torch.Size([1, 20, 13, 104, 72])
        # Wan2.2 I2V: 36 4
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    
# independent of WanVideoUnit_FunControl
class WanVideoUnit_FunControl_temporal_concat(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents", "temporal_concat"),
            output_params=("y_temporal_control_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents, temporal_concat):
        if control_video is None or not temporal_concat:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        control_video = pipe.preprocess_video(control_video)
 
        # downsample control video in spatial dimension
        # control_video shape: (B, C, T, H, W) = (1, 3, 81, 832, 480)
        B, C, T, H, W = control_video.shape
        control_video = rearrange(control_video, 'b c t h w -> (b t) c h w')
        control_video = torch.nn.functional.interpolate(control_video, scale_factor=0.5, mode='bilinear', align_corners=False)
        control_video = rearrange(control_video, '(b t) c h w -> b c t h w', b=B, t=T)

        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        return {"y_temporal_control_latents": control_latents}


# ============================================================================
# Pose limb connection definitions for warp_limbs feature
# Keypoint layout: body(18) + face(68) + hand_r(21) + hand_l(21) = 128
# ============================================================================

# Body limb connections (0-indexed, from DWPose/OpenPose convention)
# Original 1-indexed limbSeq: [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],
#   [10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]
# Note: [3,17] and [6,18] (shoulder-ear) are excluded to match draw_bodypose which only draws 17 limbs.
BODY_LIMB_CONNECTIONS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# Hand limb connections (0-indexed within each hand, 21 keypoints per hand)
HAND_LIMB_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Face has no limb connections (only points are drawn)

# Face keypoint indices (0-indexed within the 68 face keypoints) that are SKIPPED
# during pose drawing in draw_facepose_aligned. These correspond to:
#   - 0~16: jaw contour (17 points)
#   - 27~35: nose bridge (9 points)
# Total: 26 face keypoints skipped, 42 face keypoints drawn.
# This matches the SKIP_IDX logic in opensora/dataset/utils.py draw_facepose_aligned.
FACE_SKIP_IDX = set(range(0, 17)) | set(range(27, 36))


# Score threshold used by score-based keypoint filtering.
# Matches the threshold used in pose visualization (draw_keypoints_debug
# with --save_full_brightness_pose --body_score_threshold 0.3).
_SCORE_FILTER_THRESHOLD = 0.3


def _apply_score_filter(key_points, score_threshold=_SCORE_FILTER_THRESHOLD, face_skip=False):
    """Zero out validity (3rd channel) for keypoints whose score < threshold.

    This mirrors the filtering in ``draw_keypoints_debug`` with
    ``--save_full_brightness_pose --body_score_threshold 0.3``:
    - Body joints/sticks:   score < th → not drawn
    - Hand joints/edges:    score < th → not drawn
    - Face points:          score < th → not drawn; when ``face_skip`` is True,
      also always zero out FACE_SKIP_IDX (jaw contour 0-16, nose bridge 27-35).

    Requires key_points to have raw scores in the 3rd channel (pose3-style input).
    After filtering, the 3rd channel is converted to binary 0/1 so that
    downstream warp logic (which checks ``> 0``) works correctly.

    This is a pure preprocessing step that is completely independent of any
    fix_missing_warp logic. It can be composed freely with any warp variant
    (plain DirectWarp / V1 / V2 / V3).

    Args:
        key_points: (T, N, 3) tensor/array of keypoints.
        score_threshold: threshold below which a keypoint is dropped.
        face_skip: if True, also zero out FACE_SKIP_IDX (jaw contour + nose bridge)
                   regardless of their scores, matching pose drawing behavior.
    """
    num_body = 18
    num_face = 68
    num_hand = 21

    if isinstance(key_points, torch.Tensor):
        kp = key_points.clone()
    else:
        kp = torch.tensor(key_points, dtype=torch.float32)

    # kp shape: (T, N, 3)  where channel-2 = raw score
    scores = kp[:, :, 2].clone()

    # --- Body (0 .. num_body-1): score < th → 0 ---
    body_mask = scores[:, :num_body] < score_threshold
    kp[:, :num_body, 2][body_mask] = 0.0

    # --- Face (num_body .. num_body+num_face-1): score < th → 0;
    #     plus FACE_SKIP_IDX always 0 only when face_skip is enabled ---
    face_start = num_body
    face_end = num_body + num_face
    face_mask = scores[:, face_start:face_end] < score_threshold
    kp[:, face_start:face_end, 2][face_mask] = 0.0
    if face_skip:
        for local_idx in FACE_SKIP_IDX:
            global_idx = num_body + local_idx
            if global_idx < kp.shape[1]:
                kp[:, global_idx, 2] = 0.0

    # --- Hand R (face_end .. face_end+num_hand-1): score < th → 0 ---
    hand_r_start = face_end
    hand_r_end = face_end + num_hand
    if hand_r_end <= kp.shape[1]:
        hand_r_mask = scores[:, hand_r_start:hand_r_end] < score_threshold
        kp[:, hand_r_start:hand_r_end, 2][hand_r_mask] = 0.0

    # --- Hand L (hand_r_end .. hand_r_end+num_hand-1): score < th → 0 ---
    hand_l_start = hand_r_end
    hand_l_end = hand_r_end + num_hand
    if hand_l_end <= kp.shape[1]:
        hand_l_mask = scores[:, hand_l_start:hand_l_end] < score_threshold
        kp[:, hand_l_start:hand_l_end, 2][hand_l_mask] = 0.0

    # Convert remaining non-zero scores to binary 1.0 for warp compatibility
    kp[:, :, 2] = (kp[:, :, 2] > 0).float()

    return kp


def _get_face_skip_global_indices(num_body=18, num_face=68):
    """Return a set of global keypoint indices that should be skipped during warp.
    
    These are the face keypoints that are NOT drawn in pose visualization
    (jaw contour + nose bridge), converted from face-local indices to global indices.
    
    Global keypoint layout: [body(0..num_body-1), face(num_body..num_body+num_face-1), hand_r, hand_l]
    """
    return {num_body + local_idx for local_idx in FACE_SKIP_IDX}


def _get_all_limb_connections(num_body=18, num_face=68, num_hand_r=21, num_hand_l=21):
    """Build a list of (idx_a, idx_b) pairs in the global keypoint index space.
    
    Returns:
        list of (int, int): pairs of global keypoint indices that should be connected.
    """
    connections = []
    # Body connections (already in global 0-indexed space)
    for a, b in BODY_LIMB_CONNECTIONS:
        if a < num_body and b < num_body:
            connections.append((a, b))
    
    # Right hand connections (offset by num_body + num_face)
    hand_r_offset = num_body + num_face
    for a, b in HAND_LIMB_CONNECTIONS:
        connections.append((hand_r_offset + a, hand_r_offset + b))
    
    # Left hand connections (offset by num_body + num_face + num_hand_r)
    hand_l_offset = num_body + num_face + num_hand_r
    for a, b in HAND_LIMB_CONNECTIONS:
        connections.append((hand_l_offset + a, hand_l_offset + b))
    
    return connections


def _interpolate_limb_points(x0, y0, x1, y1, step=1.0):
    """Generate interpolated integer points along a line segment between two points.
    
    Args:
        x0, y0: start point (float, in latent space)
        x1, y1: end point (float, in latent space)
        step: approximate step size in latent pixels between interpolated points
    
    Returns:
        list of (int_x, int_y, frac): interpolated points with fraction along the segment.
            frac is in [0, 1] where 0 = start point, 1 = end point.
    """
    dx = x1 - x0
    dy = y1 - y0
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-6:
        return []
    
    num_steps = max(1, int(dist / step))
    points = []
    for i in range(1, num_steps):  # Skip endpoints (they are keypoints, already warped)
        frac = i / num_steps
        px = x0 + dx * frac
        py = y0 + dy * frac
        points.append((int(round(px)), int(round(py)), frac))
    return points


def _warp_limbs_on_latent(y_warped, first_frame_latent, 
                          ref_x, ref_y, ref_validity,
                          curr_x, curr_y, curr_validity,
                          t_latent, H_latent, W_latent, radius=1,
                          num_body=18, num_face=68, num_hand_r=21, num_hand_l=21):
    """Warp latent along limb connections by interpolating between keypoint pairs.
    
    For each limb connection (a, b), if both endpoints are valid in both reference
    and current frame, interpolate points along the limb in both frames and copy
    the corresponding latent from the first frame to the current frame.
    
    Args:
        y_warped: [1, C, T_latent, H_latent, W_latent] tensor being modified in-place
        first_frame_latent: [1, 16, 1, H_latent, W_latent] first frame image latent
        ref_x, ref_y, ref_validity: [num_kps] reference frame keypoint positions
        curr_x, curr_y, curr_validity: [num_kps] current frame keypoint positions
        t_latent: current latent time index
        H_latent, W_latent: latent spatial dimensions
        radius: neighborhood radius for copying
        num_body, num_face, num_hand_r, num_hand_l: keypoint counts per category
    """
    num_kps = ref_x.shape[0]
    connections = _get_all_limb_connections(num_body, num_face, num_hand_r, num_hand_l)
    
    for idx_a, idx_b in connections:
        if idx_a >= num_kps or idx_b >= num_kps:
            continue
        
        # Both endpoints must be valid in both reference and current frame
        if ref_validity[idx_a] <= 0 or ref_validity[idx_b] <= 0:
            continue
        if curr_validity[idx_a] <= 0 or curr_validity[idx_b] <= 0:
            continue
        
        # Get reference limb endpoints (float, in latent space)
        ref_ax = ref_x[idx_a].item()
        ref_ay = ref_y[idx_a].item()
        ref_bx = ref_x[idx_b].item()
        ref_by = ref_y[idx_b].item()
        
        # Get current limb endpoints (float, in latent space)
        curr_ax = curr_x[idx_a].item()
        curr_ay = curr_y[idx_a].item()
        curr_bx = curr_x[idx_b].item()
        curr_by = curr_y[idx_b].item()
        
        # Interpolate points along the reference limb
        ref_points = _interpolate_limb_points(ref_ax, ref_ay, ref_bx, ref_by, step=1.0)
        
        for src_px_f, src_py_f, frac in ref_points:
            # Compute corresponding destination point on the current limb
            dst_px_f = curr_ax + (curr_bx - curr_ax) * frac
            dst_py_f = curr_ay + (curr_by - curr_ay) * frac
            dst_px = int(round(dst_px_f))
            dst_py = int(round(dst_py_f))
            
            # Copy neighborhood from first frame to current frame
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    s_py, s_px = src_py_f + dy, src_px_f + dx
                    d_py, d_px = dst_py + dy, dst_px + dx
                    
                    if not (0 <= s_px < W_latent and 0 <= s_py < H_latent):
                        continue
                    if not (0 <= d_px < W_latent and 0 <= d_py < H_latent):
                        continue
                    
                    y_warped[:, 4:, t_latent, d_py, d_px] = first_frame_latent[:, :, 0, int(s_py), int(s_px)]


def _visualize_warp_process(output_dir, kps_x, kps_y, kps_v, H_latent, W_latent, T_latent, warp_limbs_enabled=False):
    """Visualize the latent warp process by showing ref-frame vs current-frame keypoint positions.
    
    Generates:
    1. warp_keypoints.mp4 / warp_keypoints_t*.png: Per-frame dual-color keypoint diagram.
       - RED dots = reference (first frame) keypoint positions (warp source).
       - GREEN dots = current frame keypoint positions (warp destination).
       - Thin gray line connects the same keypoint across frames to show displacement.
       - If warp_limbs_enabled, also shows limb interpolation points:
         CYAN dots = ref limb positions, YELLOW dots = current limb positions.
    2. warp_coverage.mp4 / warp_coverage_t*.png: Per-frame heatmap showing which latent pixels are covered by warp.
       - Bright = covered by warp (keypoint neighborhood), Dark = not covered (needs generation).
       - Distinguishes keypoint warp (green) vs limb warp (cyan) coverage.
    
    Args:
        output_dir: directory to save visualizations
        kps_x: [T_latent, num_kps] x-coordinates in latent space
        kps_y: [T_latent, num_kps] y-coordinates in latent space
        kps_v: [T_latent, num_kps] validity flags
        H_latent, W_latent: latent spatial dimensions
        T_latent: number of latent frames
        warp_limbs_enabled: whether limb warp is active
    """
    import os
    import numpy as np
    try:
        import cv2
        import imageio
    except ImportError:
        print("[Sanity Check] cv2 or imageio not available, skipping warp visualization")
        return

    scale = 8
    H_vis, W_vis = H_latent * scale, W_latent * scale
    radius = 1  # same as warp radius in _warp_latent_by_keypoints

    ref_x = kps_x[0]  # [num_kps]
    ref_y = kps_y[0]
    ref_v = kps_v[0]
    num_kps = kps_x.shape[1]

    # Determine keypoint category boundaries (body=18, face=68, hand_r=21, hand_l=21)
    body_end = min(18, num_kps)
    face_end = min(18 + 68, num_kps)
    hand_r_end = min(18 + 68 + 21, num_kps)

    all_limb_connections = _get_all_limb_connections()

    # Colors (BGR for cv2)
    COLOR_REF = (0, 0, 255)        # Red: reference frame keypoint position
    COLOR_CURR = (0, 255, 0)       # Green: current frame keypoint position
    COLOR_LINE = (80, 80, 80)      # Gray: displacement line connecting same keypoint
    COLOR_LIMB_REF = (200, 200, 0) # Cyan: reference limb interpolation point
    COLOR_LIMB_CURR = (0, 255, 255)# Yellow: current limb interpolation point

    # ============================================================
    # 1. Dual-color keypoint visualization (ref vs current)
    # ============================================================
    kp_frames = []
    for t in range(T_latent):
        canvas = np.zeros((H_vis, W_vis, 3), dtype=np.uint8)

        # Draw grid lines for latent pixel boundaries (subtle)
        for gy in range(0, H_vis, scale):
            cv2.line(canvas, (0, gy), (W_vis, gy), (20, 20, 20), 1)
        for gx in range(0, W_vis, scale):
            cv2.line(canvas, (gx, 0), (gx, H_vis), (20, 20, 20), 1)

        if t == 0:
            # First frame: ref and current are the same, just show one set of points
            for kp_idx in range(num_kps):
                if ref_v[kp_idx] <= 0:
                    continue
                cx = int(round(ref_x[kp_idx])) * scale + scale // 2
                cy = int(round(ref_y[kp_idx])) * scale + scale // 2
                cv2.circle(canvas, (cx, cy), max(3, scale // 3), COLOR_REF, -1)
                # Draw warp neighborhood (radius=1 means 3x3 block)
                bx = int(round(ref_x[kp_idx])) * scale
                by = int(round(ref_y[kp_idx])) * scale
                cv2.rectangle(canvas, (bx - radius * scale, by - radius * scale),
                              (bx + (radius + 1) * scale, by + (radius + 1) * scale),
                              COLOR_REF, 1)
        else:
            curr_x = kps_x[t]
            curr_y = kps_y[t]
            curr_v = kps_v[t]

            # --- Draw limb interpolation points first (behind keypoints) ---
            if warp_limbs_enabled:
                for idx_a, idx_b in all_limb_connections:
                    if idx_a >= num_kps or idx_b >= num_kps:
                        continue
                    if ref_v[idx_a] <= 0 or ref_v[idx_b] <= 0:
                        continue
                    if curr_v[idx_a] <= 0 or curr_v[idx_b] <= 0:
                        continue

                    r_ax, r_ay = ref_x[idx_a], ref_y[idx_a]
                    r_bx, r_by = ref_x[idx_b], ref_y[idx_b]
                    c_ax, c_ay = curr_x[idx_a], curr_y[idx_a]
                    c_bx, c_by = curr_x[idx_b], curr_y[idx_b]

                    ref_points = _interpolate_limb_points(r_ax, r_ay, r_bx, r_by, step=1.0)
                    for src_px_f, src_py_f, frac in ref_points:
                        dst_px_f = c_ax + (c_bx - c_ax) * frac
                        dst_py_f = c_ay + (c_by - c_ay) * frac
                        ref_vis_x = int(round(src_px_f)) * scale + scale // 2
                        ref_vis_y = int(round(src_py_f)) * scale + scale // 2
                        cur_vis_x = int(round(dst_px_f)) * scale + scale // 2
                        cur_vis_y = int(round(dst_py_f)) * scale + scale // 2
                        # Gray line connecting ref and current limb point
                        cv2.line(canvas, (ref_vis_x, ref_vis_y), (cur_vis_x, cur_vis_y), COLOR_LINE, 1)
                        # Cyan dot = ref limb position
                        cv2.circle(canvas, (ref_vis_x, ref_vis_y), 2, COLOR_LIMB_REF, -1)
                        # Yellow dot = current limb position
                        cv2.circle(canvas, (cur_vis_x, cur_vis_y), 2, COLOR_LIMB_CURR, -1)

            # --- Draw keypoints (on top of limb points) ---
            for kp_idx in range(num_kps):
                if ref_v[kp_idx] <= 0 or curr_v[kp_idx] <= 0:
                    continue

                src_cx = int(round(ref_x[kp_idx])) * scale + scale // 2
                src_cy = int(round(ref_y[kp_idx])) * scale + scale // 2
                dst_cx = int(round(curr_x[kp_idx])) * scale + scale // 2
                dst_cy = int(round(curr_y[kp_idx])) * scale + scale // 2

                # Gray line connecting ref and current position of the same keypoint
                cv2.line(canvas, (src_cx, src_cy), (dst_cx, dst_cy), COLOR_LINE, 1)
                # Red dot = reference (first frame) position
                cv2.circle(canvas, (src_cx, src_cy), max(3, scale // 3), COLOR_REF, -1)
                # Green dot = current frame position
                cv2.circle(canvas, (dst_cx, dst_cy), max(3, scale // 3), COLOR_CURR, -1)

        # Add frame label and legend
        label = f"Warp Keypoints t_latent={t}/{T_latent}"
        cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        legend_y = H_vis - 75
        cv2.circle(canvas, (10, legend_y + 5), 4, COLOR_REF, -1)
        cv2.putText(canvas, "ref frame (warp src)", (25, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.circle(canvas, (10, legend_y + 20), 4, COLOR_CURR, -1)
        cv2.putText(canvas, "curr frame (warp dst)", (25, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.line(canvas, (5, legend_y + 35), (15, legend_y + 35), COLOR_LINE, 1)
        cv2.putText(canvas, "displacement", (25, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        if warp_limbs_enabled:
            cv2.circle(canvas, (10, legend_y + 50), 3, COLOR_LIMB_REF, -1)
            cv2.putText(canvas, "limb ref", (25, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            cv2.circle(canvas, (80, legend_y + 50), 3, COLOR_LIMB_CURR, -1)
            cv2.putText(canvas, "limb curr", (95, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        kp_frames.append(canvas)

    if kp_frames:
        video_path = os.path.join(output_dir, "warp_keypoints.mp4")
        imageio.mimsave(video_path, kp_frames, fps=4)
        for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
            if idx < len(kp_frames):
                img_path = os.path.join(output_dir, f"warp_keypoints_t{idx}.png")
                imageio.imwrite(img_path, kp_frames[idx])

    # ============================================================
    # 2. Warp coverage visualization
    # ============================================================
    coverage_frames = []
    for t in range(T_latent):
        # Track which latent pixels are covered by warp
        kp_coverage = np.zeros((H_latent, W_latent), dtype=bool)
        limb_coverage = np.zeros((H_latent, W_latent), dtype=bool)

        if t == 0:
            # First frame is the reference, mark all valid keypoint neighborhoods
            for kp_idx in range(num_kps):
                if ref_v[kp_idx] <= 0:
                    continue
                px = int(round(ref_x[kp_idx]))
                py = int(round(ref_y[kp_idx]))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dst_py, dst_px = py + dy, px + dx
                        if 0 <= dst_px < W_latent and 0 <= dst_py < H_latent:
                            kp_coverage[dst_py, dst_px] = True
        else:
            curr_x = kps_x[t]
            curr_y = kps_y[t]
            curr_v = kps_v[t]

            # Keypoint warp coverage (destination positions)
            for kp_idx in range(num_kps):
                if ref_v[kp_idx] <= 0 or curr_v[kp_idx] <= 0:
                    continue
                dst_px = int(round(curr_x[kp_idx]))
                dst_py = int(round(curr_y[kp_idx]))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        d_py, d_px = dst_py + dy, dst_px + dx
                        if 0 <= d_px < W_latent and 0 <= d_py < H_latent:
                            kp_coverage[d_py, d_px] = True

            # Limb warp coverage (destination positions)
            if warp_limbs_enabled:
                for idx_a, idx_b in all_limb_connections:
                    if idx_a >= num_kps or idx_b >= num_kps:
                        continue
                    if ref_v[idx_a] <= 0 or ref_v[idx_b] <= 0:
                        continue
                    if curr_v[idx_a] <= 0 or curr_v[idx_b] <= 0:
                        continue
                    r_ax, r_ay = ref_x[idx_a], ref_y[idx_a]
                    r_bx, r_by = ref_x[idx_b], ref_y[idx_b]
                    c_ax, c_ay = curr_x[idx_a], curr_y[idx_a]
                    c_bx, c_by = curr_x[idx_b], curr_y[idx_b]
                    ref_points = _interpolate_limb_points(r_ax, r_ay, r_bx, r_by, step=1.0)
                    for src_px_f, src_py_f, frac in ref_points:
                        dst_px_f = c_ax + (c_bx - c_ax) * frac
                        dst_py_f = c_ay + (c_by - c_ay) * frac
                        dst_px = int(round(dst_px_f))
                        dst_py = int(round(dst_py_f))
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                d_py, d_px = dst_py + dy, dst_px + dx
                                if 0 <= d_px < W_latent and 0 <= d_py < H_latent:
                                    limb_coverage[d_py, d_px] = True

        # Render coverage map
        canvas = np.zeros((H_vis, W_vis, 3), dtype=np.uint8)
        for py in range(H_latent):
            for px in range(W_latent):
                y1, y2 = py * scale, (py + 1) * scale
                x1, x2 = px * scale, (px + 1) * scale
                if kp_coverage[py, px] and limb_coverage[py, px]:
                    canvas[y1:y2, x1:x2] = [0, 255, 200]  # Teal: both kp and limb
                elif kp_coverage[py, px]:
                    canvas[y1:y2, x1:x2] = [0, 255, 0]    # Green: keypoint warp
                elif limb_coverage[py, px]:
                    canvas[y1:y2, x1:x2] = [0, 200, 200]  # Cyan: limb warp only
                else:
                    canvas[y1:y2, x1:x2] = [24, 24, 24]    # Dark: not covered

        # Compute coverage stats
        total_pixels = H_latent * W_latent
        kp_count = int(kp_coverage.sum())
        limb_count = int(limb_coverage.sum())
        combined = int((kp_coverage | limb_coverage).sum())

        # Add frame label and stats
        label = f"Warp Coverage t_latent={t}/{T_latent}  kp={kp_count} limb={limb_count} total={combined}/{total_pixels} ({combined/total_pixels:.1%})"
        cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        # Legend
        legend_y = H_vis - 55
        cv2.rectangle(canvas, (5, legend_y), (15, legend_y + 10), (0, 255, 0), -1)
        cv2.putText(canvas, "kp warp", (20, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.rectangle(canvas, (5, legend_y + 15), (15, legend_y + 25), (0, 200, 200), -1)
        cv2.putText(canvas, "limb warp", (20, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.rectangle(canvas, (5, legend_y + 30), (15, legend_y + 40), (24, 24, 24), -1)
        cv2.putText(canvas, "not covered", (20, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        coverage_frames.append(canvas)

    if coverage_frames:
        video_path = os.path.join(output_dir, "warp_coverage.mp4")
        imageio.mimsave(video_path, coverage_frames, fps=4)
        for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
            if idx < len(coverage_frames):
                img_path = os.path.join(output_dir, f"warp_coverage_t{idx}.png")
                imageio.imwrite(img_path, coverage_frames[idx])

    print(f"[Sanity Check] Warp process visualizations saved to {output_dir}/")
    print(f"  - warp_keypoints.mp4 + warp_keypoints_t*.png (red=ref, green=current keypoint positions)")
    print(f"  - warp_coverage.mp4 + warp_coverage_t*.png (latent pixel coverage map)")


def _visualize_warp_keypoints_pose_style(output_path, kps_x, kps_y, kps_v, H_latent, W_latent, T_latent,
                                          height, width, num_frames, warp_limbs_enabled=False):
    """Visualize warped keypoints in pose-style (pixel space), similar to infer_preprocess_yyx.py.
    
    This draws keypoints and limb connections on a black background in pixel space,
    using the ACTUAL warped keypoint positions from the latent warp process.
    The visualization is consistent with the warp results because it reads from the
    same kps_x/kps_y/kps_v data that was used for warping.
    
    For each latent frame, the corresponding pixel frames are generated by upscaling
    the latent keypoint coordinates to pixel space.
    
    Output: a video file at output_path with T_pixel frames (expanded from T_latent).
    
    Args:
        output_path: path to save the output video (e.g., "output_vis_warp_kps.mp4")
        kps_x: [T_latent, num_kps] x-coordinates in latent space
        kps_y: [T_latent, num_kps] y-coordinates in latent space
        kps_v: [T_latent, num_kps] validity flags
        H_latent, W_latent: latent spatial dimensions
        T_latent: number of latent frames
        height, width: pixel-space dimensions
        num_frames: number of pixel frames
        warp_limbs_enabled: whether limb warp is active (affects which connections are drawn)
    """
    import os
    import numpy as np
    try:
        import cv2
        import imageio
    except ImportError:
        print("[vis_warp_keypoints] cv2 or imageio not available, skipping")
        return

    num_kps = kps_x.shape[1]

    # Keypoint category boundaries (body=18, face=68, hand_r=21, hand_l=21)
    body_end = min(18, num_kps)
    face_end = min(18 + 68, num_kps)
    hand_r_end = min(18 + 68 + 21, num_kps)

    # Limb connections
    all_limb_connections = _get_all_limb_connections()

    # Body limb colors (matching OpenPose style - each limb has a different color)
    # Using a colorful palette for body limbs
    body_limb_colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170),
    ]
    # Hand limb color
    hand_limb_color = (255, 255, 0)  # Yellow for hands

    # Latent-to-pixel scale factors
    # Latent coords are in [0, W_latent-1] and [0, H_latent-1]
    # Pixel coords should be in [0, width-1] and [0, height-1]
    # patch_size_h = 16, patch_size_w = 16 (latent pixel = 16x16 pixel patch)
    scale_x = width / W_latent
    scale_y = height / H_latent

    # Generate frames for each latent timestep
    # Each latent frame corresponds to multiple pixel frames (temporal compression factor = 4)
    # But for visualization, we generate one frame per latent timestep
    vis_frames = []
    for t in range(T_latent):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # --- Draw limb connections first (behind keypoints) ---
        for conn_idx, (idx_a, idx_b) in enumerate(all_limb_connections):
            if idx_a >= num_kps or idx_b >= num_kps:
                continue
            if kps_v[t, idx_a] <= 0 or kps_v[t, idx_b] <= 0:
                continue

            # Convert latent coords to pixel coords
            ax = int(round(kps_x[t, idx_a] * scale_x + scale_x / 2))
            ay = int(round(kps_y[t, idx_a] * scale_y + scale_y / 2))
            bx = int(round(kps_x[t, idx_b] * scale_x + scale_x / 2))
            by = int(round(kps_y[t, idx_b] * scale_y + scale_y / 2))

            # Determine color based on connection type
            if idx_a < body_end and idx_b < body_end:
                # Body limb - use colorful palette
                color_idx = min(conn_idx, len(body_limb_colors) - 1)
                color = body_limb_colors[color_idx]
                thickness = 3
            else:
                # Hand limb
                color = hand_limb_color
                thickness = 2

            cv2.line(canvas, (ax, ay), (bx, by), color, thickness)

        # --- Draw keypoints on top ---
        for kp_idx in range(num_kps):
            if kps_v[t, kp_idx] <= 0:
                continue

            # Convert latent coords to pixel coords
            px = int(round(kps_x[t, kp_idx] * scale_x + scale_x / 2))
            py = int(round(kps_y[t, kp_idx] * scale_y + scale_y / 2))

            if px < 0 or px >= width or py < 0 or py >= height:
                continue

            if kp_idx < body_end:
                color = (255, 0, 0)    # Red for body joints
                radius = 5
            elif kp_idx < face_end:
                color = (0, 100, 255)  # Blue for face
                radius = 2
            else:
                color = (255, 255, 0)  # Yellow for hands
                radius = 3

            cv2.circle(canvas, (px, py), radius, color, -1)

        vis_frames.append(canvas)

    # Expand latent frames to pixel frames (each latent frame -> 4 pixel frames, except first -> 1)
    # Temporal compression: T_pixel = (T_latent - 1) * 4 + 1
    expanded_frames = []
    for t in range(T_latent):
        if t == 0:
            expanded_frames.append(vis_frames[t])
        else:
            # Repeat each non-first latent frame 4 times
            for _ in range(4):
                expanded_frames.append(vis_frames[t])

    # Trim to match actual num_frames
    expanded_frames = expanded_frames[:num_frames]

    # Save as video
    if expanded_frames:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        imageio.mimsave(output_path, expanded_frames, fps=15)
        print(f"[vis_warp_keypoints] Saved pose-style warp keypoints visualization to {output_path}")


class WanVideoUnit_DirectWarp(PipelineUnit):
    # Warp first frame latent following Wan-Move based on keypoints

    def __init__(self):
        super().__init__(
            input_params=("control_video", "input_image", "key_points", "num_frames", "height", "width", "clip_feature", "y", "latents", "fix_missing_warp", "fix_missing_warp_v2", "fix_missing_warp_v3", "score_filter", "warp_limbs", "face_skip", "vis_warp_keypoints", "vis_warp_keypoints_path"),
            output_params=("clip_feature", "y"),
            onload_model_names=()  # No need to load VAE - control_video is already encoded
        )

    def process(self, pipe: WanVideoPipeline, control_video, input_image, key_points, num_frames, height, width, clip_feature, y, latents, fix_missing_warp, fix_missing_warp_v2=False, fix_missing_warp_v3=False, score_filter=False, warp_limbs=False, face_skip=False, vis_warp_keypoints=False, vis_warp_keypoints_path=""):
        # Skip if any fix_missing variant is enabled (handled by dedicated units)
        if fix_missing_warp or fix_missing_warp_v2 or fix_missing_warp_v3:
            return {}
        # Only proceed when control_video, input_image, and key_points are all present
        if control_video is None or input_image is None or key_points is None:
            return {}

        # Apply score-based filtering (independent, orthogonal to fix_missing_warp).
        # When enabled, filters low-confidence keypoints BEFORE warp.
        if score_filter:
            key_points = _apply_score_filter(key_points, face_skip=face_skip)

        # After WanVideoUnit_FunControl, y has shape [1, C_total, T_latent, H_latent, W_latent]
        # where C_total = 20 (mask + image_latent) + 16 (control_latent) = 36
        # y[:, :20] is original y (mask + image_latent)
        # y[:, 20:] is control_latent (already encoded by WanVideoUnit_FunControl)
        
        # Separate control_latent from y
        control_latent_dim = 16  # control_latent has 16 channels
        y_original_dim = 20     # mask (4) + image_latent (16) = 20
        
        # Extract control_latent and original y from the concatenated tensor
        control_latents = y[:, y_original_dim:]  # [1, 16, T_latent, H_latent, W_latent]
        y_original = y[:, :y_original_dim]       # [1, 20, T_latent, H_latent, W_latent]
        
        if clip_feature is not None and y_original is not None:
            # ip2v pipeline: y_original has shape [1, 20, T_latent, H_latent, W_latent]
            # y_original[:, :4] is mask, y_original[:, 4:20] is image latent
            # clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            
            # Apply keypoint-based warping to image latent (only modifies latent, NOT mask)
            y_warped = self.warp_latent_by_keypoints(y_original, key_points, height, width, num_frames, warp_limbs=warp_limbs, face_skip=face_skip)

            # Pose-style keypoints warp visualization (if enabled, only on rank 0 to avoid duplicate writes under USP)
            if vis_warp_keypoints and hasattr(self, '_kps_latent_x_ds'):
                import torch.distributed as dist
                is_main = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
                if is_main:
                    _, _, T_lat, H_lat, W_lat = y_warped.shape
                    kps_x_np = self._kps_latent_x_ds.detach().cpu().numpy()
                    kps_y_np = self._kps_latent_y_ds.detach().cpu().numpy()
                    kps_v_np = self._kps_validity_ds.detach().cpu().numpy()
                    vis_path = vis_warp_keypoints_path if vis_warp_keypoints_path else os.path.join("output", "vis_warp_keypoints.mp4")
                    _visualize_warp_keypoints_pose_style(
                        vis_path, kps_x_np, kps_y_np, kps_v_np,
                        H_lat, W_lat, T_lat, height, width, num_frames,
                        warp_limbs_enabled=warp_limbs,
                    )

            # DEBUG: Print value range and diff positions between non-first frames and first frame
            if 0:
                _, C, T_latent, H_latent, W_latent = y_warped.shape
                image_latent_warped = y_warped[:, 4:]  # [1, 16, T_latent, H_latent, W_latent]
                first_frame = image_latent_warped[:, :, 0]  # [1, 16, H_latent, W_latent]
                print(f"[DEBUG after warp] y_warped shape: {y_warped.shape}, T_latent={T_latent}, H_latent={H_latent}, W_latent={W_latent}")
                print(f"[DEBUG after warp] First frame (t=0): min={first_frame.min().item():.4f}, max={first_frame.max().item():.4f}, mean={first_frame.mean().item():.4f}")
                if T_latent > 1:
                    non_first_frames = image_latent_warped[:, :, 1:]  # [1, 16, T_latent-1, H_latent, W_latent]
                    print(f"[DEBUG after warp] Non-first frames (t=1~{T_latent-1}): min={non_first_frames.min().item():.4f}, max={non_first_frames.max().item():.4f}, mean={non_first_frames.mean().item():.4f}")
                    # Find positions where non-first frames differ from first frame
                    # Expand first_frame to compare with each non-first frame: [1, 16, H, W] -> [1, 16, T-1, H, W]
                    first_frame_expanded = first_frame.unsqueeze(2).expand_as(non_first_frames)
                    # Check across all 16 channels - if any channel has difference at a position, consider it different
                    diff_mask = ((non_first_frames - first_frame_expanded).abs() > 1e-6).any(dim=1)  # [1, T_latent-1, H_latent, W_latent]
                    diff_indices = torch.nonzero(diff_mask[0], as_tuple=False)  # [N, 3] where each row is [t, h, w]
                    total_diff_count = diff_indices.shape[0]
                    print(f"[DEBUG after warp] Diff positions count (non-first vs first): {total_diff_count}")
                    if total_diff_count > 0:
                        # Print first 20 diff positions as (t, h, w, H_latent, W_latent)
                        num_to_print = min(20, total_diff_count)
                        print(f"[DEBUG after warp] First {num_to_print} diff positions (t_latent, y_latent, x_latent, H_latent, W_latent):")
                        for i in range(num_to_print):
                            t_idx, h_idx, w_idx = diff_indices[i].tolist()
                            # t_idx is relative to non-first frames, so actual t = t_idx + 1
                            print(f"    ({t_idx + 1}, {h_idx}, {w_idx}, {H_latent}, {W_latent})")

                assert 0
                
                # check y_warped.shape and y_original.shape, min and max
                # print(y_warped.shape, y_original.shape, y_warped.min(), y_warped.max(), y_original.min(), y_original.max());assert 0
                # torch.Size([1, 20, 21, 104, 60]) torch.Size([1, 20, 21, 104, 60]) tensor(-2.2344, device='cuda:0', dtype=torch.bfloat16) tensor(2.7812, device='cuda:0', dtype=torch.bfloat16) tensor(-2.2344, device='cuda:0', dtype=torch.bfloat16) tensor(2.7812, device='cuda:0', dtype=torch.bfloat16)
                
        else:
            assert 0, "Unexpected case in WanVideoUnit_DirectWarp"
            y_warped = y_original

        return {"clip_feature": clip_feature, "y": y_warped}

    
    def warp_latent_by_keypoints(self, y, key_points, height, width, num_frames, warp_limbs=False, face_skip=False):
        """
        Warp image latent based on keypoint movements.
        
        Args:
            y: tensor of shape [1, 20, T_latent, H_latent, W_latent]
               y[:, :4] is mask, y[:, 4:20] is image latent (16 channels)
            key_points: tensor of shape [T_pixel, N, 3] where
               [:, :, 0] is x (normalized 0~1), [:, :, 1] is y (normalized 0~1), [:, :, 2] is validity
            height: pixel height
            width: pixel width
            num_frames: number of pixel frames
            warp_limbs: if True, also warp along limb connections between keypoints
        
        Returns:
            y_warped: tensor with warped image latent
        """
        # Spatial compression factor: 8x, Temporal compression factor: 4x
        spatial_scale = 8
        temporal_scale = 4
        
        # Get latent dimensions
        _, C, T_latent, H_latent, W_latent = y.shape
        
        # y[:, :4] is mask (NOT modified), y[:, 4:] is image latent (will be warped)
        # Only modify latent, DO NOT modify mask
        image_latent = y[:, 4:]  # [1, 16, T_latent, H_latent, W_latent]
        
        # Get first frame latent (reference frame)
        first_frame_latent = image_latent[:, :, 0:1]  # [1, 16, 1, H_latent, W_latent]
        
        # Convert key_points to tensor if needed, keep on same device as y
        if isinstance(key_points, torch.Tensor):
            kps = key_points.to(device=y.device).float()  # Convert to float32 for numerical stability
        else:
            kps = torch.tensor(key_points, dtype=torch.float32, device=y.device)
        
        T_pixel = kps.shape[0]  # Number of pixel-level frames
        num_kps = kps.shape[1]  # Number of keypoints (typically 18)
        
        # Convert pixel coordinates to latent coordinates
        # key_points[:, :, 0] is x (along width), [:, :, 1] is y (along height)
        # They are normalized (0~1), need to convert to latent space
        kps_latent_x = kps[:, :, 0] * W_latent  # [T_pixel, num_kps]
        kps_latent_y = kps[:, :, 1] * H_latent  # [T_pixel, num_kps]
        kps_validity = kps[:, :, 2] if kps.shape[2] >= 3 else torch.ones(T_pixel, num_kps, device=y.device)  # [T_pixel, num_kps]
        
        # Temporal downsampling using tensor operations
        # Pre-allocate tensors for downsampled keypoints
        kps_latent_x_downsampled = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        kps_latent_y_downsampled = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        kps_validity_downsampled = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        
        for t_latent in range(T_latent):
            # Map latent frame index to pixel frame indices
            if t_latent == 0:
                start_idx, end_idx = 0, 1
            else:
                start_idx = 1 + (t_latent - 1) * temporal_scale
                end_idx = min(start_idx + temporal_scale, T_pixel)
            
            if start_idx >= T_pixel:
                start_idx, end_idx = T_pixel - 1, T_pixel
            
            # Get frames for this latent timestep
            frame_x = kps_latent_x[start_idx:end_idx]  # [num_frames_in_group, num_kps]
            frame_y = kps_latent_y[start_idx:end_idx]  # [num_frames_in_group, num_kps]
            frame_validity = kps_validity[start_idx:end_idx]  # [num_frames_in_group, num_kps]
            
            # Create mask for valid points (validity > 0)
            valid_mask = frame_validity > 0  # [num_frames_in_group, num_kps]
            
            # Calculate weighted average (using validity as mask)
            valid_count = valid_mask.float().sum(dim=0).clamp(min=1)  # [num_kps], avoid div by zero
            x_sum = (frame_x * valid_mask.float()).sum(dim=0)  # [num_kps]
            y_sum = (frame_y * valid_mask.float()).sum(dim=0)  # [num_kps]
            
            kps_latent_x_downsampled[t_latent] = x_sum / valid_count
            kps_latent_y_downsampled[t_latent] = y_sum / valid_count
            # Validity: valid if any frame in the group has valid keypoint
            kps_validity_downsampled[t_latent] = frame_validity.max(dim=0)[0]  # [num_kps]
        
        # Zero out validity for face keypoints that are skipped in pose drawing
        # (jaw contour 0-16 and nose bridge 27-35, matching draw_facepose_aligned SKIP_IDX)
        # Controlled by an independent face_skip flag (defaults to False; Shell
        # side sets its default to follow WARP_LIMBS to preserve legacy behavior).
        if face_skip:
            face_skip_global = _get_face_skip_global_indices()
            for skip_idx in face_skip_global:
                if skip_idx < num_kps:
                    kps_validity_downsampled[:, skip_idx] = 0.0

        # Get first frame keypoint positions (reference positions)
        ref_x = kps_latent_x_downsampled[0]  # [num_kps]
        ref_y = kps_latent_y_downsampled[0]  # [num_kps]
        ref_validity = kps_validity_downsampled[0]  # [num_kps]
        
        # Create output tensor (clone y to preserve mask - mask will NOT be modified)
        y_warped = y.clone()
        
        # Warp radius for neighborhood copying
        radius = 1
        
        # For each subsequent latent frame, warp the first frame latent based on keypoint movement
        for t_latent in range(1, T_latent):
            curr_x = kps_latent_x_downsampled[t_latent]  # [num_kps]
            curr_y = kps_latent_y_downsampled[t_latent]  # [num_kps]
            curr_validity = kps_validity_downsampled[t_latent]  # [num_kps]
            
            # Find valid keypoint pairs (both reference and current are valid)
            valid_pair_mask = (ref_validity > 0) & (curr_validity > 0)  # [num_kps]
            
            if not valid_pair_mask.any():
                continue
            
            # Get valid keypoint indices
            valid_indices = torch.where(valid_pair_mask)[0]
            
            for kp_idx in valid_indices:
                # Get reference and target positions (round to integer)
                src_x = int(ref_x[kp_idx].round().item())
                src_y = int(ref_y[kp_idx].round().item())
                dst_x = int(curr_x[kp_idx].round().item())
                dst_y = int(curr_y[kp_idx].round().item())
                
                # Copy neighborhood from first frame to current frame at warped position
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        src_py, src_px = src_y + dy, src_x + dx
                        dst_py, dst_px = dst_y + dy, dst_x + dx
                        
                        # Bounds check
                        if not (0 <= src_px < W_latent and 0 <= src_py < H_latent):
                            continue
                        if not (0 <= dst_px < W_latent and 0 <= dst_py < H_latent):
                            continue
                        
                        # Copy from first frame latent to current frame at warped position
                        y_warped[:, 4:, t_latent, dst_py, dst_px] = first_frame_latent[:, :, 0, src_py, src_px]
        
        # Warp along limb connections if enabled
        if warp_limbs:
            for t_latent in range(1, T_latent):
                curr_x = kps_latent_x_downsampled[t_latent]
                curr_y = kps_latent_y_downsampled[t_latent]
                curr_validity = kps_validity_downsampled[t_latent]
                _warp_limbs_on_latent(
                    y_warped, first_frame_latent,
                    ref_x, ref_y, ref_validity,
                    curr_x, curr_y, curr_validity,
                    t_latent, H_latent, W_latent, radius=radius,
                )

        # Store downsampled keypoints for reuse (e.g., vis_warp_keypoints)
        self._kps_latent_x_ds = kps_latent_x_downsampled
        self._kps_latent_y_ds = kps_latent_y_downsampled
        self._kps_validity_ds = kps_validity_downsampled

        return y_warped


class WanVideoUnit_DirectWarpFixMissing(PipelineUnit):
    """Warp first frame latent with fix-missing logic:
    1. Modify y[:, :4] mask for non-first frames:
       - 1 where keypoints exist in both first and current frame (warpable, no generation needed)
       - 0 for non-keypoint areas (keep original, need generation)
       - -1 where keypoints exist in current frame but missing in first frame (mark missing)
    2. Generate a 4-channel keypoint index embedding map and append to y,
       mapping each keypoint index to a learned 4-channel embedding at its spatial position.
    
    Note: num_keypoints is dynamically read from key_points.shape[1] (e.g. 128 for DWPose, 134 for SMPL+DWPose).
    """

    def __init__(self):
        super().__init__(
            input_params=("control_video", "input_image", "key_points", "num_frames", "height", "width", "clip_feature", "y", "latents", "fix_missing_warp", "score_filter", "sanity_check", "sanity_check_data_id", "warp_limbs", "face_skip", "vis_warp_keypoints", "vis_warp_keypoints_path"),
            output_params=("clip_feature", "y"),
            onload_model_names=()
        )

    def process(self, pipe: WanVideoPipeline, control_video, input_image, key_points, num_frames, height, width, clip_feature, y, latents, fix_missing_warp, score_filter=False, sanity_check=False, sanity_check_data_id=None, warp_limbs=False, face_skip=False, vis_warp_keypoints=False, vis_warp_keypoints_path=""):
        # Only proceed when fix_missing_warp is enabled and all required inputs are present
        if not fix_missing_warp:
            return {}
        if control_video is None or input_image is None or key_points is None:
            return {}

        # Apply score-based filtering (independent, orthogonal to fix_missing_warp).
        if score_filter:
            key_points = _apply_score_filter(key_points)

        # y layout: y[:, :20] = original y (mask 4ch + image_latent 16ch), y[:, 20:] = control_latent (16ch)
        control_latent_dim = 16
        y_original_dim = 20
        control_latents = y[:, y_original_dim:]   # [1, 16, T_latent, H_latent, W_latent]
        y_original = y[:, :y_original_dim]        # [1, 20, T_latent, H_latent, W_latent]

        if clip_feature is None or y_original is None:
            return {}

        # Step 1: Warp image latent (same as DirectWarp)
        y_warped = self._warp_latent_by_keypoints(y_original, key_points, height, width, num_frames, warp_limbs=warp_limbs, face_skip=face_skip)

        # Step 2: Modify mask for missing keypoints (set to -1)
        y_warped = self._modify_mask_for_missing(y_warped, key_points, height, width, num_frames)

        # Step 3: Generate keypoint index embedding (4 channels)
        kp_index_emb = self._generate_keypoint_index_embedding(y_warped, key_points, height, width, num_frames)

        # Concatenate: y_warped (20ch) + kp_index_emb (4ch) = 24ch
        # Note: control_latents are discarded (same as WanVideoUnit_DirectWarp),
        # the extra 4 channels are for keypoint index embedding
        y_out = torch.cat([y_warped, kp_index_emb], dim=1)

        # Sanity check: visualize mask and keypoint index embedding
        if sanity_check:
            self._sanity_check_visualize(y_warped, kp_index_emb, key_points, height, width, num_frames, sanity_check_data_id=sanity_check_data_id, warp_limbs=warp_limbs)

        # Pose-style keypoints warp visualization (if enabled, only on rank 0 to avoid duplicate writes under USP)
        if vis_warp_keypoints and hasattr(self, '_kps_latent_x_ds'):
            import torch.distributed as dist
            is_main = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
            if is_main:
                _, _, T_lat, H_lat, W_lat = y_warped.shape
                kps_x_np = self._kps_latent_x_ds.detach().cpu().numpy()
                kps_y_np = self._kps_latent_y_ds.detach().cpu().numpy()
                kps_v_np = self._kps_validity_ds.detach().cpu().numpy()
                vis_path = vis_warp_keypoints_path if vis_warp_keypoints_path else os.path.join("output", "vis_warp_keypoints.mp4")
                _visualize_warp_keypoints_pose_style(
                    vis_path, kps_x_np, kps_y_np, kps_v_np,
                    H_lat, W_lat, T_lat, height, width, num_frames,
                    warp_limbs_enabled=warp_limbs,
                )

        return {"clip_feature": clip_feature, "y": y_out}

    def _warp_latent_by_keypoints(self, y, key_points, height, width, num_frames, warp_limbs=False, face_skip=False):
        """Warp image latent based on keypoint movements (same logic as WanVideoUnit_DirectWarp)."""
        spatial_scale = 8
        temporal_scale = 4
        _, C, T_latent, H_latent, W_latent = y.shape

        image_latent = y[:, 4:]  # [1, 16, T_latent, H_latent, W_latent]
        first_frame_latent = image_latent[:, :, 0:1]

        if isinstance(key_points, torch.Tensor):
            kps = key_points.to(device=y.device).float()
        else:
            kps = torch.tensor(key_points, dtype=torch.float32, device=y.device)

        T_pixel = kps.shape[0]
        num_kps = kps.shape[1]

        kps_latent_x = kps[:, :, 0] * W_latent
        kps_latent_y = kps[:, :, 1] * H_latent
        kps_validity = kps[:, :, 2] if kps.shape[2] >= 3 else torch.ones(T_pixel, num_kps, device=y.device)

        # Temporal downsampling
        kps_latent_x_ds = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        kps_latent_y_ds = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        kps_validity_ds = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)

        for t_latent in range(T_latent):
            if t_latent == 0:
                start_idx, end_idx = 0, 1
            else:
                start_idx = 1 + (t_latent - 1) * temporal_scale
                end_idx = min(start_idx + temporal_scale, T_pixel)
            if start_idx >= T_pixel:
                start_idx, end_idx = T_pixel - 1, T_pixel

            frame_x = kps_latent_x[start_idx:end_idx]
            frame_y = kps_latent_y[start_idx:end_idx]
            frame_validity = kps_validity[start_idx:end_idx]
            valid_mask = frame_validity > 0
            valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
            kps_latent_x_ds[t_latent] = (frame_x * valid_mask.float()).sum(dim=0) / valid_count
            kps_latent_y_ds[t_latent] = (frame_y * valid_mask.float()).sum(dim=0) / valid_count
            kps_validity_ds[t_latent] = frame_validity.max(dim=0)[0]

        # Zero out validity for face keypoints that are skipped in pose drawing
        # (jaw contour 0-16 and nose bridge 27-35, matching draw_facepose_aligned SKIP_IDX)
        # Controlled by an independent face_skip flag.
        if face_skip:
            face_skip_global = _get_face_skip_global_indices()
            for skip_idx in face_skip_global:
                if skip_idx < num_kps:
                    kps_validity_ds[:, skip_idx] = 0.0

        ref_x = kps_latent_x_ds[0]
        ref_y = kps_latent_y_ds[0]
        ref_validity = kps_validity_ds[0]

        y_warped = y.clone()
        radius = 1

        for t_latent in range(1, T_latent):
            curr_x = kps_latent_x_ds[t_latent]
            curr_y = kps_latent_y_ds[t_latent]
            curr_validity = kps_validity_ds[t_latent]
            valid_pair_mask = (ref_validity > 0) & (curr_validity > 0)
            if not valid_pair_mask.any():
                continue
            valid_indices = torch.where(valid_pair_mask)[0]
            for kp_idx in valid_indices:
                src_x = int(ref_x[kp_idx].round().item())
                src_y = int(ref_y[kp_idx].round().item())
                dst_x = int(curr_x[kp_idx].round().item())
                dst_y = int(curr_y[kp_idx].round().item())
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        src_py, src_px = src_y + dy, src_x + dx
                        dst_py, dst_px = dst_y + dy, dst_x + dx
                        if not (0 <= src_px < W_latent and 0 <= src_py < H_latent):
                            continue
                        if not (0 <= dst_px < W_latent and 0 <= dst_py < H_latent):
                            continue
                        y_warped[:, 4:, t_latent, dst_py, dst_px] = first_frame_latent[:, :, 0, src_py, src_px]

        # Warp along limb connections if enabled
        if warp_limbs:
            for t_latent in range(1, T_latent):
                curr_x = kps_latent_x_ds[t_latent]
                curr_y = kps_latent_y_ds[t_latent]
                curr_validity = kps_validity_ds[t_latent]
                _warp_limbs_on_latent(
                    y_warped, first_frame_latent,
                    ref_x, ref_y, ref_validity,
                    curr_x, curr_y, curr_validity,
                    t_latent, H_latent, W_latent, radius=radius,
                )

        # Store downsampled keypoints for reuse
        self._kps_latent_x_ds = kps_latent_x_ds
        self._kps_latent_y_ds = kps_latent_y_ds
        self._kps_validity_ds = kps_validity_ds

        return y_warped

    def _modify_mask_for_missing(self, y, key_points, height, width, num_frames):
        """Modify y[:, :4] mask for non-first frames based on keypoint visibility.
        
        For non-first frames:
        - mask = 1 at positions where keypoints exist in BOTH first and current frame (warpable)
        - mask = 0 at non-keypoint positions (need generation, same as original)
        - mask = -1 at positions where keypoints exist in current frame but MISSING in first frame
        """
        _, C, T_latent, H_latent, W_latent = y.shape
        radius = 1

        kps_latent_x_ds = self._kps_latent_x_ds
        kps_latent_y_ds = self._kps_latent_y_ds
        kps_validity_ds = self._kps_validity_ds

        ref_validity = kps_validity_ds[0]  # [num_kps]

        for t_latent in range(1, T_latent):
            curr_x = kps_latent_x_ds[t_latent]
            curr_y = kps_latent_y_ds[t_latent]
            curr_validity = kps_validity_ds[t_latent]
            num_kps = curr_x.shape[0]

            for kp_idx in range(num_kps):
                if curr_validity[kp_idx] <= 0:
                    continue  # Current frame keypoint not valid, skip

                dst_x = int(curr_x[kp_idx].round().item())
                dst_y = int(curr_y[kp_idx].round().item())

                if ref_validity[kp_idx] > 0:
                    # Both first and current frame have this keypoint -> mark as warpable (1)
                    mask_val = 1.0
                else:
                    # Current frame has keypoint but first frame doesn't -> mark as missing (-1)
                    mask_val = -1.0

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dst_py, dst_px = dst_y + dy, dst_x + dx
                        if 0 <= dst_px < W_latent and 0 <= dst_py < H_latent:
                            y[:, :4, t_latent, dst_py, dst_px] = mask_val

        return y

    def _generate_keypoint_index_embedding(self, y, key_points, height, width, num_frames):
        """Generate a 4-channel keypoint index embedding map.
        
        For each frame, at each keypoint's spatial position in latent space,
        encode the keypoint index (0~N-1) as a 4-channel embedding using
        a simple sinusoidal encoding scheme.
        
        Returns: tensor [1, 4, T_latent, H_latent, W_latent]
        """
        _, C, T_latent, H_latent, W_latent = y.shape
        radius = 1

        kps_latent_x_ds = self._kps_latent_x_ds
        kps_latent_y_ds = self._kps_latent_y_ds
        kps_validity_ds = self._kps_validity_ds
        num_kps = kps_latent_x_ds.shape[1]

        # Create 4-channel index embedding for each keypoint index
        # Use sinusoidal encoding: for index i, embed as [sin(i*f1), cos(i*f1), sin(i*f2), cos(i*f2)]
        # where f1, f2 are different frequencies to ensure unique embeddings
        indices = torch.arange(num_kps, dtype=torch.float32, device=y.device)
        # Normalize indices to [0, 1] range for stable encoding
        indices_norm = indices / max(num_kps - 1, 1)
        freq1 = 2.0 * torch.pi
        freq2 = 4.0 * torch.pi
        kp_embeddings = torch.stack([
            torch.sin(indices_norm * freq1),
            torch.cos(indices_norm * freq1),
            torch.sin(indices_norm * freq2),
            torch.cos(indices_norm * freq2),
        ], dim=1)  # [num_kps, 4]

        # Initialize embedding map with zeros
        emb_map = torch.zeros(1, 4, T_latent, H_latent, W_latent, dtype=y.dtype, device=y.device)

        for t_latent in range(T_latent):
            curr_x = kps_latent_x_ds[t_latent]
            curr_y = kps_latent_y_ds[t_latent]
            curr_validity = kps_validity_ds[t_latent]

            for kp_idx in range(num_kps):
                if curr_validity[kp_idx] <= 0:
                    continue

                px = int(curr_x[kp_idx].round().item())
                py = int(curr_y[kp_idx].round().item())
                emb = kp_embeddings[kp_idx]  # [4]

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dst_py, dst_px = py + dy, px + dx
                        if 0 <= dst_px < W_latent and 0 <= dst_py < H_latent:
                            emb_map[0, :, t_latent, dst_py, dst_px] = emb.to(dtype=y.dtype)

        return emb_map

    def _sanity_check_visualize(self, y_warped, kp_index_emb, key_points, height, width, num_frames, sanity_check_data_id=None, warp_limbs=False):
        """Visualize mask modification and keypoint index embedding for sanity check.
        
        Saves visualization images/videos to sanity_check_output/data_{data_id}/fix_missing_warp/ directory.
        
        Visualizations:
        1. Mask visualization (y[:, :4]): color-coded per-frame images showing:
           - Green (1.0): warpable positions (keypoints in both first and current frame)
           - Black (0.0): non-keypoint areas (need generation)
           - Red (-1.0): missing positions (keypoints in current but not first frame)
           - Blue: first frame mask (original, typically all 1s)
        2. Keypoint index embedding visualization (4 channels): 
           - Each channel visualized as a heatmap
           - Combined RGB visualization mapping 4 channels to color
        3. Warped image latent difference visualization
        """
        import os
        import numpy as np

        if sanity_check_data_id is not None:
            output_dir = os.path.join("sanity_check_output", f"data_{sanity_check_data_id}", "fix_missing_warp")
        else:
            output_dir = os.path.join("sanity_check_output", "fix_missing_warp")
        os.makedirs(output_dir, exist_ok=True)

        _, C, T_latent, H_latent, W_latent = y_warped.shape
        mask = y_warped[0, :4].detach().cpu().float().numpy()       # [4, T_latent, H_latent, W_latent]
        emb = kp_index_emb[0].detach().cpu().float().numpy()        # [4, T_latent, H_latent, W_latent]
        image_latent = y_warped[0, 4:].detach().cpu().float().numpy()  # [16, T_latent, H_latent, W_latent]

        # Also get keypoint info for overlay
        kps_x = self._kps_latent_x_ds.detach().cpu().numpy()  # [T_latent, num_kps]
        kps_y = self._kps_latent_y_ds.detach().cpu().numpy()
        kps_v = self._kps_validity_ds.detach().cpu().numpy()
        ref_validity = kps_v[0]

        try:
            import cv2
            import imageio

            # ============================================================
            # 1. Mask visualization: color-coded video
            # ============================================================
            mask_vis_frames = []
            for t in range(T_latent):
                # Use channel 0 as representative (all 4 channels have same mask values)
                mask_ch0 = mask[0, t]  # [H_latent, W_latent]

                # Create RGB canvas: upscale for better visibility
                scale = 8
                H_vis, W_vis = H_latent * scale, W_latent * scale
                canvas = np.zeros((H_vis, W_vis, 3), dtype=np.uint8)

                # Color code: iterate over latent pixels
                for py in range(H_latent):
                    for px in range(W_latent):
                        val = mask_ch0[py, px]
                        y1, y2 = py * scale, (py + 1) * scale
                        x1, x2 = px * scale, (px + 1) * scale
                        if val > 0.5:
                            # Warpable (1.0) -> Green
                            canvas[y1:y2, x1:x2] = [0, 255, 0]
                        elif val < -0.5:
                            # Missing (-1.0) -> Red
                            canvas[y1:y2, x1:x2] = [255, 0, 0]
                        else:
                            # Non-keypoint (0.0) -> Dark gray (to distinguish from pure black border)
                            if t == 0:
                                # First frame: show original mask value
                                # Original mask is typically 1.0 for first frame -> Blue
                                canvas[y1:y2, x1:x2] = [64, 64, 255]
                            else:
                                canvas[y1:y2, x1:x2] = [32, 32, 32]

                # Overlay keypoint positions as small dots
                for kp_idx in range(kps_x.shape[1]):
                    if kps_v[t, kp_idx] <= 0:
                        continue
                    cx = int(round(kps_x[t, kp_idx])) * scale + scale // 2
                    cy = int(round(kps_y[t, kp_idx])) * scale + scale // 2
                    if 0 <= cx < W_vis and 0 <= cy < H_vis:
                        # Draw keypoint marker
                        if t == 0 or ref_validity[kp_idx] > 0:
                            color = (255, 255, 255)  # White dot for valid/warpable
                        else:
                            color = (255, 128, 0)    # Orange dot for missing-in-first-frame
                        cv2.circle(canvas, (cx, cy), max(2, scale // 3), color, -1)
                        # Add keypoint index text
                        cv2.putText(canvas, str(kp_idx), (cx + 3, cy - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

                # Overlay limb connections as lines between connected keypoints
                all_limb_connections = _get_all_limb_connections()
                for (idx_a, idx_b) in all_limb_connections:
                    if idx_a >= kps_x.shape[1] or idx_b >= kps_x.shape[1]:
                        continue
                    if kps_v[t, idx_a] <= 0 or kps_v[t, idx_b] <= 0:
                        continue
                    ax = int(round(kps_x[t, idx_a])) * scale + scale // 2
                    ay = int(round(kps_y[t, idx_a])) * scale + scale // 2
                    bx = int(round(kps_x[t, idx_b])) * scale + scale // 2
                    by = int(round(kps_y[t, idx_b])) * scale + scale // 2
                    # Use cyan for body limbs, yellow for hand limbs
                    if idx_a < 18 and idx_b < 18:
                        line_color = (0, 255, 255)  # Cyan for body
                    else:
                        line_color = (255, 255, 0)  # Yellow for hands
                    cv2.line(canvas, (ax, ay), (bx, by), line_color, 1)

                # Add frame label and legend
                label = f"t_latent={t}/{T_latent}"
                cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # Legend
                legend_y = H_vis - 60
                cv2.rectangle(canvas, (5, legend_y), (15, legend_y + 10), (0, 255, 0), -1)
                cv2.putText(canvas, "warpable(1)", (20, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.rectangle(canvas, (5, legend_y + 15), (15, legend_y + 25), (255, 0, 0), -1)
                cv2.putText(canvas, "missing(-1)", (20, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.rectangle(canvas, (5, legend_y + 30), (15, legend_y + 40), (32, 32, 32), -1)
                cv2.putText(canvas, "need_gen(0)", (20, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                mask_vis_frames.append(canvas)

            # Save mask visualization as video and individual frames
            if mask_vis_frames:
                video_path = os.path.join(output_dir, "mask_visualization.mp4")
                imageio.mimsave(video_path, mask_vis_frames, fps=4)
                # Save first few frames as images
                for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
                    if idx < len(mask_vis_frames):
                        img_path = os.path.join(output_dir, f"mask_t{idx}.png")
                        imageio.imwrite(img_path, mask_vis_frames[idx])

            # ============================================================
            # 2. Keypoint index embedding visualization
            # ============================================================
            emb_vis_frames = []
            for t in range(T_latent):
                scale = 8
                H_vis, W_vis = H_latent * scale, W_latent * scale

                # Visualize each of the 4 embedding channels as a separate heatmap
                channel_imgs = []
                ch_names = ["sin(f1)", "cos(f1)", "sin(f2)", "cos(f2)"]
                for ch in range(4):
                    ch_data = emb[ch, t]  # [H_latent, W_latent], range [-1, 1]
                    # Normalize to [0, 255]: map [-1, 1] -> [0, 255]
                    ch_norm = ((ch_data + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
                    ch_upscaled = cv2.resize(ch_norm, (W_vis, H_vis), interpolation=cv2.INTER_NEAREST)
                    ch_colored = cv2.applyColorMap(ch_upscaled, cv2.COLORMAP_JET)
                    ch_colored = cv2.cvtColor(ch_colored, cv2.COLOR_BGR2RGB)
                    # Mark zero regions (no keypoint) as dark
                    zero_mask = (np.abs(ch_data) < 1e-6)
                    zero_mask_up = cv2.resize(zero_mask.astype(np.uint8), (W_vis, H_vis), interpolation=cv2.INTER_NEAREST).astype(bool)
                    ch_colored[zero_mask_up] = [16, 16, 16]
                    # Add channel label
                    cv2.putText(ch_colored, ch_names[ch], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    channel_imgs.append(ch_colored)

                # Arrange 4 channels in 2x2 grid
                top_row = np.concatenate([channel_imgs[0], channel_imgs[1]], axis=1)
                bot_row = np.concatenate([channel_imgs[2], channel_imgs[3]], axis=1)
                grid = np.concatenate([top_row, bot_row], axis=0)

                # Add frame label
                cv2.putText(grid, f"KP Index Emb t_latent={t}/{T_latent}", (5, grid.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                emb_vis_frames.append(grid)

            if emb_vis_frames:
                video_path = os.path.join(output_dir, "kp_index_embedding.mp4")
                imageio.mimsave(video_path, emb_vis_frames, fps=4)
                for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
                    if idx < len(emb_vis_frames):
                        img_path = os.path.join(output_dir, f"kp_index_emb_t{idx}.png")
                        imageio.imwrite(img_path, emb_vis_frames[idx])

            # ============================================================
            # 3. Combined keypoint index map (decode embedding back to index)
            # ============================================================
            kp_index_vis_frames = []
            num_kps = kps_x.shape[1]
            # Generate color palette for keypoint indices
            np.random.seed(42)
            palette = np.random.randint(50, 255, size=(num_kps, 3), dtype=np.uint8)

            for t in range(T_latent):
                scale = 8
                H_vis, W_vis = H_latent * scale, W_latent * scale
                canvas = np.zeros((H_vis, W_vis, 3), dtype=np.uint8)

                for kp_idx in range(num_kps):
                    if kps_v[t, kp_idx] <= 0:
                        continue
                    px = int(round(kps_x[t, kp_idx]))
                    py = int(round(kps_y[t, kp_idx]))
                    color = tuple(int(c) for c in palette[kp_idx])
                    radius = 1
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            dst_py, dst_px = py + dy, px + dx
                            if 0 <= dst_px < W_latent and 0 <= dst_py < H_latent:
                                y1, y2 = dst_py * scale, (dst_py + 1) * scale
                                x1, x2 = dst_px * scale, (dst_px + 1) * scale
                                canvas[y1:y2, x1:x2] = color

                    # Draw keypoint index label
                    cx = px * scale + scale // 2
                    cy = py * scale + scale // 2
                    if 0 <= cx < W_vis and 0 <= cy < H_vis:
                        cv2.putText(canvas, str(kp_idx), (cx + 3, cy - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

                cv2.putText(canvas, f"KP Index Map t_latent={t}/{T_latent}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                kp_index_vis_frames.append(canvas)

            if kp_index_vis_frames:
                video_path = os.path.join(output_dir, "kp_index_map.mp4")
                imageio.mimsave(video_path, kp_index_vis_frames, fps=4)
                for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
                    if idx < len(kp_index_vis_frames):
                        img_path = os.path.join(output_dir, f"kp_index_map_t{idx}.png")
                        imageio.imwrite(img_path, kp_index_vis_frames[idx])

            # ============================================================
            # 4. Summary statistics
            # ============================================================
            summary = {
                "latent_shape": f"T={T_latent}, H={H_latent}, W={W_latent}",
                "num_keypoints": int(num_kps),
                "pixel_height": int(height),
                "pixel_width": int(width),
                "num_frames": int(num_frames),
            }
            # Per-frame mask statistics
            mask_stats = []
            for t in range(T_latent):
                m = mask[0, t]
                warpable = int((m > 0.5).sum())
                missing = int((m < -0.5).sum())
                need_gen = int((np.abs(m) <= 0.5).sum())
                total = H_latent * W_latent
                mask_stats.append({
                    "t_latent": t,
                    "warpable(1)": warpable,
                    "missing(-1)": missing,
                    "need_gen(0)": int(need_gen),
                    "total": total,
                    "warpable_ratio": f"{warpable / total:.2%}",
                    "missing_ratio": f"{missing / total:.2%}",
                })
            summary["mask_stats_per_frame"] = mask_stats

            # Per-frame keypoint validity
            kp_stats = []
            for t in range(T_latent):
                valid_count = int((kps_v[t] > 0).sum())
                if t > 0:
                    both_valid = int(((kps_v[0] > 0) & (kps_v[t] > 0)).sum())
                    curr_only = int(((kps_v[0] <= 0) & (kps_v[t] > 0)).sum())
                    ref_only = int(((kps_v[0] > 0) & (kps_v[t] <= 0)).sum())
                else:
                    both_valid = valid_count
                    curr_only = 0
                    ref_only = 0
                kp_stats.append({
                    "t_latent": t,
                    "valid": valid_count,
                    "both_valid": both_valid,
                    "curr_only(missing)": curr_only,
                    "ref_only(lost)": ref_only,
                })
            summary["keypoint_stats_per_frame"] = kp_stats

            import json
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"[Sanity Check] Fix-missing warp visualizations saved to {output_dir}/")
            print(f"  - mask_visualization.mp4 + mask_t*.png (color-coded mask: green=warpable, red=missing, dark=need_gen)")
            print(f"  - kp_index_embedding.mp4 + kp_index_emb_t*.png (4-channel sinusoidal embedding heatmaps)")
            print(f"  - kp_index_map.mp4 + kp_index_map_t*.png (keypoint index color map)")
            print(f"  - summary.json (statistics)")

            # Warp process visualization (arrows + coverage)
            _visualize_warp_process(output_dir, kps_x, kps_y, kps_v, H_latent, W_latent, T_latent, warp_limbs_enabled=warp_limbs)

        except Exception as e:
            print(f"[Sanity Check] Error during fix_missing_warp visualization: {e}")
            import traceback
            traceback.print_exc()


class WanVideoUnit_DirectWarpFixMissingV2(PipelineUnit):
    """Warp first frame latent with fix-missing V2 logic:
    
    Differences from V1 (WanVideoUnit_DirectWarpFixMissing):
    1. Mask keeps original first-frame/non-first-frame global 0/1 logic (no -1 marking).
    2. For keypoints that exist in current frame but are MISSING in first frame,
       generate a learned embedding (mapped from keypoint index to 16ch, same dim as latent)
       and use it to fill the warped image latent at those positions.
    3. Still generates 4-channel keypoint index embedding for all valid keypoints.
    
    Output: y_warped (20ch) + kp_index_emb (4ch) = 24ch (same as V1)
    """

    def __init__(self):
        super().__init__(
            input_params=("control_video", "input_image", "key_points", "num_frames", "height", "width", "clip_feature", "y", "latents", "fix_missing_warp_v2", "score_filter", "sanity_check", "sanity_check_data_id", "warp_limbs", "face_skip", "vis_warp_keypoints", "vis_warp_keypoints_path"),
            output_params=("clip_feature", "y"),
            onload_model_names=()
        )

    def process(self, pipe: WanVideoPipeline, control_video, input_image, key_points, num_frames, height, width, clip_feature, y, latents, fix_missing_warp_v2=False, score_filter=False, sanity_check=False, sanity_check_data_id=None, warp_limbs=False, face_skip=False, vis_warp_keypoints=False, vis_warp_keypoints_path=""):
        # Only proceed when fix_missing_warp_v2 is enabled and all required inputs are present
        if not fix_missing_warp_v2:
            return {}
        if control_video is None or input_image is None or key_points is None:
            return {}

        # Apply score-based filtering (independent, orthogonal to fix_missing_warp_v2).
        if score_filter:
            key_points = _apply_score_filter(key_points)

        # y layout: y[:, :20] = original y (mask 4ch + image_latent 16ch), y[:, 20:] = control_latent (16ch)
        control_latent_dim = 16
        y_original_dim = 20
        control_latents = y[:, y_original_dim:]   # [1, 16, T_latent, H_latent, W_latent]
        y_original = y[:, :y_original_dim]        # [1, 20, T_latent, H_latent, W_latent]

        if clip_feature is None or y_original is None:
            return {}

        # Step 1: Warp image latent (same as DirectWarp) - mask is NOT modified
        y_warped = self._warp_latent_by_keypoints(y_original, key_points, height, width, num_frames, warp_limbs=warp_limbs, face_skip=face_skip)

        # Step 2: Fill missing keypoint positions with learned embeddings (16ch)
        # For positions where current frame has keypoint but first frame doesn't,
        # replace the image latent (which would be unwarped/default) with a learned embedding
        # from pipe.dit.kp_index_embedding_16ch
        y_warped = self._fill_missing_with_embedding(pipe, y_warped, key_points, height, width, num_frames)

        # Step 3: Generate keypoint index embedding (4 channels) using learned embedding
        # from pipe.dit.kp_index_embedding_4ch
        kp_index_emb = self._generate_keypoint_index_embedding(pipe, y_warped, key_points, height, width, num_frames)

        # Concatenate: y_warped (20ch) + kp_index_emb (4ch) = 24ch
        y_out = torch.cat([y_warped, kp_index_emb], dim=1)

        # Sanity check visualization
        if sanity_check:
            self._sanity_check_visualize(y_warped, kp_index_emb, key_points, height, width, num_frames, sanity_check_data_id=sanity_check_data_id, warp_limbs=warp_limbs)

        # Pose-style keypoints warp visualization (if enabled, only on rank 0 to avoid duplicate writes under USP)
        if vis_warp_keypoints and hasattr(self, '_kps_latent_x_ds'):
            import torch.distributed as dist
            is_main = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
            if is_main:
                _, _, T_lat, H_lat, W_lat = y_warped.shape
                kps_x_np = self._kps_latent_x_ds.detach().cpu().numpy()
                kps_y_np = self._kps_latent_y_ds.detach().cpu().numpy()
                kps_v_np = self._kps_validity_ds.detach().cpu().numpy()
                vis_path = vis_warp_keypoints_path if vis_warp_keypoints_path else os.path.join("output", "vis_warp_keypoints.mp4")
                _visualize_warp_keypoints_pose_style(
                    vis_path, kps_x_np, kps_y_np, kps_v_np,
                    H_lat, W_lat, T_lat, height, width, num_frames,
                    warp_limbs_enabled=warp_limbs,
                )

        return {"clip_feature": clip_feature, "y": y_out}

    def _warp_latent_by_keypoints(self, y, key_points, height, width, num_frames, warp_limbs=False, face_skip=False):
        """Warp image latent based on keypoint movements (same logic as WanVideoUnit_DirectWarp).
        Mask (y[:, :4]) is NOT modified - keeps original 0/1 global logic."""
        spatial_scale = 8
        temporal_scale = 4
        _, C, T_latent, H_latent, W_latent = y.shape

        image_latent = y[:, 4:]  # [1, 16, T_latent, H_latent, W_latent]
        first_frame_latent = image_latent[:, :, 0:1]

        if isinstance(key_points, torch.Tensor):
            kps = key_points.to(device=y.device).float()
        else:
            kps = torch.tensor(key_points, dtype=torch.float32, device=y.device)

        T_pixel = kps.shape[0]
        num_kps = kps.shape[1]

        kps_latent_x = kps[:, :, 0] * W_latent
        kps_latent_y = kps[:, :, 1] * H_latent
        kps_validity = kps[:, :, 2] if kps.shape[2] >= 3 else torch.ones(T_pixel, num_kps, device=y.device)

        # Temporal downsampling
        kps_latent_x_ds = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        kps_latent_y_ds = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)
        kps_validity_ds = torch.zeros(T_latent, num_kps, device=y.device, dtype=torch.float32)

        for t_latent in range(T_latent):
            if t_latent == 0:
                start_idx, end_idx = 0, 1
            else:
                start_idx = 1 + (t_latent - 1) * temporal_scale
                end_idx = min(start_idx + temporal_scale, T_pixel)
            if start_idx >= T_pixel:
                start_idx, end_idx = T_pixel - 1, T_pixel

            frame_x = kps_latent_x[start_idx:end_idx]
            frame_y = kps_latent_y[start_idx:end_idx]
            frame_validity = kps_validity[start_idx:end_idx]
            valid_mask = frame_validity > 0
            valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
            kps_latent_x_ds[t_latent] = (frame_x * valid_mask.float()).sum(dim=0) / valid_count
            kps_latent_y_ds[t_latent] = (frame_y * valid_mask.float()).sum(dim=0) / valid_count
            kps_validity_ds[t_latent] = frame_validity.max(dim=0)[0]

        # Zero out validity for face keypoints that are skipped in pose drawing
        # (jaw contour 0-16 and nose bridge 27-35, matching draw_facepose_aligned SKIP_IDX)
        # Controlled by an independent face_skip flag.
        if face_skip:
            face_skip_global = _get_face_skip_global_indices()
            for skip_idx in face_skip_global:
                if skip_idx < num_kps:
                    kps_validity_ds[:, skip_idx] = 0.0

        ref_x = kps_latent_x_ds[0]
        ref_y = kps_latent_y_ds[0]
        ref_validity = kps_validity_ds[0]

        y_warped = y.clone()
        radius = 1

        for t_latent in range(1, T_latent):
            curr_x = kps_latent_x_ds[t_latent]
            curr_y = kps_latent_y_ds[t_latent]
            curr_validity = kps_validity_ds[t_latent]
            valid_pair_mask = (ref_validity > 0) & (curr_validity > 0)
            if not valid_pair_mask.any():
                continue
            valid_indices = torch.where(valid_pair_mask)[0]
            for kp_idx in valid_indices:
                src_x = int(ref_x[kp_idx].round().item())
                src_y = int(ref_y[kp_idx].round().item())
                dst_x = int(curr_x[kp_idx].round().item())
                dst_y = int(curr_y[kp_idx].round().item())
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        src_py, src_px = src_y + dy, src_x + dx
                        dst_py, dst_px = dst_y + dy, dst_x + dx
                        if not (0 <= src_px < W_latent and 0 <= src_py < H_latent):
                            continue
                        if not (0 <= dst_px < W_latent and 0 <= dst_py < H_latent):
                            continue
                        y_warped[:, 4:, t_latent, dst_py, dst_px] = first_frame_latent[:, :, 0, src_py, src_px]

        # Warp along limb connections if enabled
        if warp_limbs:
            for t_latent in range(1, T_latent):
                curr_x = kps_latent_x_ds[t_latent]
                curr_y = kps_latent_y_ds[t_latent]
                curr_validity = kps_validity_ds[t_latent]
                _warp_limbs_on_latent(
                    y_warped, first_frame_latent,
                    ref_x, ref_y, ref_validity,
                    curr_x, curr_y, curr_validity,
                    t_latent, H_latent, W_latent, radius=radius,
                )

        # Store downsampled keypoints for reuse
        self._kps_latent_x_ds = kps_latent_x_ds
        self._kps_latent_y_ds = kps_latent_y_ds
        self._kps_validity_ds = kps_validity_ds

        return y_warped

    def _fill_missing_with_embedding(self, pipe, y, key_points, height, width, num_frames):
        """Fill missing keypoint positions with learned embeddings.
        
        For non-first frames, at positions where keypoints exist in current frame
        but are MISSING in first frame, replace the image latent (16ch) with a
        learned embedding from pipe.dit.kp_index_embedding_16ch.
        
        This provides the model with a learnable signal at missing positions
        instead of the default (unwarped) first-frame latent.
        """
        _, C, T_latent, H_latent, W_latent = y.shape
        radius = 1

        kps_latent_x_ds = self._kps_latent_x_ds
        kps_latent_y_ds = self._kps_latent_y_ds
        kps_validity_ds = self._kps_validity_ds
        num_kps = kps_latent_x_ds.shape[1]

        ref_validity = kps_validity_ds[0]  # [num_kps]

        # Get 16-channel learned embedding for each keypoint index from DiT model
        kp_indices = torch.arange(num_kps, dtype=torch.long, device=y.device)
        kp_embeddings_16ch = pipe.dit.kp_index_embedding_16ch(kp_indices)  # [num_kps, 16]

        for t_latent in range(1, T_latent):
            curr_x = kps_latent_x_ds[t_latent]
            curr_y = kps_latent_y_ds[t_latent]
            curr_validity = kps_validity_ds[t_latent]

            for kp_idx in range(num_kps):
                if curr_validity[kp_idx] <= 0:
                    continue  # Current frame keypoint not valid, skip
                if ref_validity[kp_idx] > 0:
                    continue  # Both frames have this keypoint, already warped, skip

                # Current frame has keypoint but first frame doesn't -> fill with embedding
                dst_x = int(curr_x[kp_idx].round().item())
                dst_y = int(curr_y[kp_idx].round().item())
                emb_16ch = kp_embeddings_16ch[kp_idx]  # [16]

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dst_py, dst_px = dst_y + dy, dst_x + dx
                        if 0 <= dst_px < W_latent and 0 <= dst_py < H_latent:
                            y[:, 4:20, t_latent, dst_py, dst_px] = emb_16ch.to(dtype=y.dtype)

        return y

    def _generate_keypoint_index_embedding(self, pipe, y, key_points, height, width, num_frames):
        """Generate a 4-channel keypoint index embedding map using learned embeddings.
        
        For each frame, at each keypoint's spatial position in latent space,
        encode the keypoint index (0~N-1) as a 4-channel learned embedding
        from pipe.dit.kp_index_embedding_4ch.
        
        Returns: tensor [1, 4, T_latent, H_latent, W_latent]
        """
        _, C, T_latent, H_latent, W_latent = y.shape
        radius = 1

        kps_latent_x_ds = self._kps_latent_x_ds
        kps_latent_y_ds = self._kps_latent_y_ds
        kps_validity_ds = self._kps_validity_ds
        num_kps = kps_latent_x_ds.shape[1]

        # Get 4-channel learned embedding for each keypoint index from DiT model
        kp_indices = torch.arange(num_kps, dtype=torch.long, device=y.device)
        kp_embeddings = pipe.dit.kp_index_embedding_4ch(kp_indices)  # [num_kps, 4]

        emb_map = torch.zeros(1, 4, T_latent, H_latent, W_latent, dtype=y.dtype, device=y.device)

        for t_latent in range(T_latent):
            curr_x = kps_latent_x_ds[t_latent]
            curr_y = kps_latent_y_ds[t_latent]
            curr_validity = kps_validity_ds[t_latent]

            for kp_idx in range(num_kps):
                if curr_validity[kp_idx] <= 0:
                    continue

                px = int(curr_x[kp_idx].round().item())
                py = int(curr_y[kp_idx].round().item())
                emb = kp_embeddings[kp_idx]  # [4]

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dst_py, dst_px = py + dy, px + dx
                        if 0 <= dst_px < W_latent and 0 <= dst_py < H_latent:
                            emb_map[0, :, t_latent, dst_py, dst_px] = emb.to(dtype=y.dtype)

        return emb_map

    def _sanity_check_visualize(self, y_warped, kp_index_emb, key_points, height, width, num_frames, sanity_check_data_id=None, warp_limbs=False):
        """Visualize V2 fix-missing warp results for sanity check.
        
        Saves visualization images/videos to sanity_check_output/data_{data_id}/fix_missing_warp_v2/ directory.
        """
        import os
        import numpy as np

        if sanity_check_data_id is not None:
            output_dir = os.path.join("sanity_check_output", f"data_{sanity_check_data_id}", "fix_missing_warp_v2")
        else:
            output_dir = os.path.join("sanity_check_output", "fix_missing_warp_v2")
        os.makedirs(output_dir, exist_ok=True)

        _, C, T_latent, H_latent, W_latent = y_warped.shape
        mask = y_warped[0, :4].detach().cpu().float().numpy()
        emb = kp_index_emb[0].detach().cpu().float().numpy()
        image_latent = y_warped[0, 4:].detach().cpu().float().numpy()

        kps_x = self._kps_latent_x_ds.detach().cpu().numpy()
        kps_y = self._kps_latent_y_ds.detach().cpu().numpy()
        kps_v = self._kps_validity_ds.detach().cpu().numpy()
        ref_validity = kps_v[0]

        try:
            import cv2
            import imageio

            # Mask visualization (V2: only 0/1, no -1)
            mask_vis_frames = []
            for t in range(T_latent):
                mask_ch0 = mask[0, t]
                scale = 8
                H_vis, W_vis = H_latent * scale, W_latent * scale
                canvas = np.zeros((H_vis, W_vis, 3), dtype=np.uint8)

                for py in range(H_latent):
                    for px in range(W_latent):
                        val = mask_ch0[py, px]
                        y1, y2 = py * scale, (py + 1) * scale
                        x1, x2 = px * scale, (px + 1) * scale
                        if val > 0.5:
                            canvas[y1:y2, x1:x2] = [0, 255, 0]  # Green: mask=1
                        else:
                            if t == 0:
                                canvas[y1:y2, x1:x2] = [64, 64, 255]  # Blue: first frame
                            else:
                                canvas[y1:y2, x1:x2] = [32, 32, 32]  # Dark: mask=0

                # Overlay keypoint positions
                for kp_idx in range(kps_x.shape[1]):
                    if kps_v[t, kp_idx] <= 0:
                        continue
                    cx = int(round(kps_x[t, kp_idx])) * scale + scale // 2
                    cy = int(round(kps_y[t, kp_idx])) * scale + scale // 2
                    if 0 <= cx < W_vis and 0 <= cy < H_vis:
                        if t == 0 or ref_validity[kp_idx] > 0:
                            color = (255, 255, 255)  # White: valid/warpable
                        else:
                            color = (255, 128, 0)    # Orange: missing-in-first-frame (filled with embedding)
                        cv2.circle(canvas, (cx, cy), max(2, scale // 3), color, -1)
                        cv2.putText(canvas, str(kp_idx), (cx + 3, cy - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

                # Overlay limb connections as lines between connected keypoints
                all_limb_connections = _get_all_limb_connections()
                for (idx_a, idx_b) in all_limb_connections:
                    if idx_a >= kps_x.shape[1] or idx_b >= kps_x.shape[1]:
                        continue
                    if kps_v[t, idx_a] <= 0 or kps_v[t, idx_b] <= 0:
                        continue
                    ax = int(round(kps_x[t, idx_a])) * scale + scale // 2
                    ay = int(round(kps_y[t, idx_a])) * scale + scale // 2
                    bx = int(round(kps_x[t, idx_b])) * scale + scale // 2
                    by = int(round(kps_y[t, idx_b])) * scale + scale // 2
                    # Use cyan for body limbs, yellow for hand limbs
                    if idx_a < 18 and idx_b < 18:
                        line_color = (0, 255, 255)  # Cyan for body
                    else:
                        line_color = (255, 255, 0)  # Yellow for hands
                    cv2.line(canvas, (ax, ay), (bx, by), line_color, 1)

                label = f"V2 t_latent={t}/{T_latent}"
                cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                legend_y = H_vis - 45
                cv2.rectangle(canvas, (5, legend_y), (15, legend_y + 10), (0, 255, 0), -1)
                cv2.putText(canvas, "mask=1", (20, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.rectangle(canvas, (5, legend_y + 15), (15, legend_y + 25), (32, 32, 32), -1)
                cv2.putText(canvas, "mask=0", (20, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.rectangle(canvas, (5, legend_y + 30), (15, legend_y + 40), (255, 128, 0), -1)
                cv2.putText(canvas, "emb_filled", (20, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                mask_vis_frames.append(canvas)

            if mask_vis_frames:
                video_path = os.path.join(output_dir, "mask_visualization.mp4")
                imageio.mimsave(video_path, mask_vis_frames, fps=4)
                for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
                    if idx < len(mask_vis_frames):
                        img_path = os.path.join(output_dir, f"mask_t{idx}.png")
                        imageio.imwrite(img_path, mask_vis_frames[idx])

            # Summary statistics
            import json
            num_kps = kps_x.shape[1]
            summary = {
                "version": "V2",
                "latent_shape": f"T={T_latent}, H={H_latent}, W={W_latent}",
                "num_keypoints": int(num_kps),
                "pixel_height": int(height),
                "pixel_width": int(width),
                "num_frames": int(num_frames),
                "description": "V2: mask keeps 0/1 global logic, missing keypoints filled with 16ch sinusoidal embedding",
            }
            kp_stats = []
            for t in range(T_latent):
                valid_count = int((kps_v[t] > 0).sum())
                if t > 0:
                    both_valid = int(((kps_v[0] > 0) & (kps_v[t] > 0)).sum())
                    curr_only = int(((kps_v[0] <= 0) & (kps_v[t] > 0)).sum())
                    ref_only = int(((kps_v[0] > 0) & (kps_v[t] <= 0)).sum())
                else:
                    both_valid = valid_count
                    curr_only = 0
                    ref_only = 0
                kp_stats.append({
                    "t_latent": t,
                    "valid": valid_count,
                    "both_valid": both_valid,
                    "curr_only(emb_filled)": curr_only,
                    "ref_only(lost)": ref_only,
                })
            summary["keypoint_stats_per_frame"] = kp_stats

            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"[Sanity Check] Fix-missing warp V2 visualizations saved to {output_dir}/")

            # Warp process visualization (arrows + coverage)
            _visualize_warp_process(output_dir, kps_x, kps_y, kps_v, H_latent, W_latent, T_latent, warp_limbs_enabled=warp_limbs)

        except Exception as e:
            print(f"[Sanity Check] Error during fix_missing_warp_v2 visualization: {e}")
            import traceback
            traceback.print_exc()


class WanVideoUnit_DirectWarpFixMissingV3(WanVideoUnit_DirectWarpFixMissingV2):
    """Warp first frame latent with fix-missing V3 logic:

    Differences from V2 (WanVideoUnit_DirectWarpFixMissingV2):
    1. Same warp logic as V2 (mask keeps 0/1, missing keypoints filled with 16ch learned embedding).
    2. Does NOT generate the 4-channel keypoint index embedding map.
       Output is y_warped (20ch) — no extra channels appended.
       This means patch_embedding does NOT need to be expanded (+4).
    3. Only requires kp_index_embedding_16ch (no kp_index_embedding_4ch needed).

    Output: y_warped (20ch) — same channel count as plain DirectWarp.
    """

    def __init__(self):
        # Call PipelineUnit.__init__ directly to set our own input_params with fix_missing_warp_v3 gate
        PipelineUnit.__init__(
            self,
            input_params=(
                "control_video", "input_image", "key_points", "num_frames",
                "height", "width", "clip_feature", "y", "latents",
                "fix_missing_warp_v3", "score_filter",
                "sanity_check", "sanity_check_data_id",
                "warp_limbs", "face_skip", "vis_warp_keypoints", "vis_warp_keypoints_path",
            ),
            output_params=("clip_feature", "y"),
            onload_model_names=(),
        )

    def process(
        self, pipe: WanVideoPipeline,
        control_video, input_image, key_points, num_frames, height, width,
        clip_feature, y, latents,
        fix_missing_warp_v3=False, score_filter=False,
        sanity_check=False, sanity_check_data_id=None,
        warp_limbs=False, face_skip=False, vis_warp_keypoints=False, vis_warp_keypoints_path="",
    ):
        # Only proceed when fix_missing_warp_v3 is enabled
        if not fix_missing_warp_v3:
            return {}
        if control_video is None or input_image is None or key_points is None:
            return {}

        # Apply score-based filtering (independent, orthogonal to fix_missing_warp_v3).
        if score_filter:
            key_points = _apply_score_filter(key_points)

        # y layout: y[:, :20] = original y (mask 4ch + image_latent 16ch), y[:, 20:] = control_latent (16ch)
        control_latent_dim = 16
        y_original_dim = 20
        control_latents = y[:, y_original_dim:]   # [1, 16, T_latent, H_latent, W_latent]
        y_original = y[:, :y_original_dim]        # [1, 20, T_latent, H_latent, W_latent]

        if clip_feature is None or y_original is None:
            return {}

        # Step 1: Warp image latent (same as DirectWarp) - mask is NOT modified
        y_warped = self._warp_latent_by_keypoints(y_original, key_points, height, width, num_frames, warp_limbs=warp_limbs, face_skip=face_skip)

        # Step 2: Fill missing keypoint positions with learned embeddings (16ch)
        # For positions where current frame has keypoint but first frame doesn't,
        # replace the image latent with a learned embedding from pipe.dit.kp_index_embedding_16ch
        y_warped = self._fill_missing_with_embedding(pipe, y_warped, key_points, height, width, num_frames)

        # NOTE: V3 does NOT do Step 3 (no 4ch keypoint index embedding generation/concat).
        # Output is y_warped (20ch) directly — no patch_embedding expansion needed.

        # Sanity check visualization (reuse V2 visualizer but pass None for kp_index_emb)
        if sanity_check:
            self._sanity_check_visualize_v3(y_warped, key_points, height, width, num_frames, sanity_check_data_id=sanity_check_data_id, warp_limbs=warp_limbs)

        # Pose-style keypoints warp visualization
        if vis_warp_keypoints and hasattr(self, '_kps_latent_x_ds'):
            import torch.distributed as dist
            is_main = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
            if is_main:
                _, _, T_lat, H_lat, W_lat = y_warped.shape
                kps_x_np = self._kps_latent_x_ds.detach().cpu().numpy()
                kps_y_np = self._kps_latent_y_ds.detach().cpu().numpy()
                kps_v_np = self._kps_validity_ds.detach().cpu().numpy()
                vis_path = vis_warp_keypoints_path if vis_warp_keypoints_path else os.path.join("output", "vis_warp_keypoints.mp4")
                _visualize_warp_keypoints_pose_style(
                    vis_path, kps_x_np, kps_y_np, kps_v_np,
                    H_lat, W_lat, T_lat, height, width, num_frames,
                    warp_limbs_enabled=warp_limbs,
                )

        return {"clip_feature": clip_feature, "y": y_warped}

    def _sanity_check_visualize_v3(self, y_warped, key_points, height, width, num_frames, sanity_check_data_id=None, warp_limbs=False):
        """Visualize V3 fix-missing warp results for sanity check.

        Similar to V2 but without kp_index_emb visualization.
        Saves to sanity_check_output/data_{data_id}/fix_missing_warp_v3/ directory.
        """
        import os
        import numpy as np

        if sanity_check_data_id is not None:
            output_dir = os.path.join("sanity_check_output", f"data_{sanity_check_data_id}", "fix_missing_warp_v3")
        else:
            output_dir = os.path.join("sanity_check_output", "fix_missing_warp_v3")
        os.makedirs(output_dir, exist_ok=True)

        _, C, T_latent, H_latent, W_latent = y_warped.shape
        mask = y_warped[0, :4].detach().cpu().float().numpy()
        image_latent = y_warped[0, 4:].detach().cpu().float().numpy()

        kps_x = self._kps_latent_x_ds.detach().cpu().numpy()
        kps_y = self._kps_latent_y_ds.detach().cpu().numpy()
        kps_v = self._kps_validity_ds.detach().cpu().numpy()
        ref_validity = kps_v[0]

        try:
            import cv2
            import imageio

            # Mask visualization (V3: same as V2, only 0/1, no -1)
            mask_vis_frames = []
            for t in range(T_latent):
                mask_ch0 = mask[0, t]
                scale = 8
                H_vis, W_vis = H_latent * scale, W_latent * scale
                canvas = np.zeros((H_vis, W_vis, 3), dtype=np.uint8)

                for py in range(H_latent):
                    for px in range(W_latent):
                        val = mask_ch0[py, px]
                        y1, y2 = py * scale, (py + 1) * scale
                        x1, x2 = px * scale, (px + 1) * scale
                        if val > 0.5:
                            canvas[y1:y2, x1:x2] = [0, 255, 0]  # Green: mask=1
                        else:
                            if t == 0:
                                canvas[y1:y2, x1:x2] = [64, 64, 255]  # Blue: first frame
                            else:
                                canvas[y1:y2, x1:x2] = [32, 32, 32]  # Dark: mask=0

                # Overlay keypoint positions
                for kp_idx in range(kps_x.shape[1]):
                    if kps_v[t, kp_idx] <= 0:
                        continue
                    cx = int(round(kps_x[t, kp_idx])) * scale + scale // 2
                    cy = int(round(kps_y[t, kp_idx])) * scale + scale // 2
                    if 0 <= cx < W_vis and 0 <= cy < H_vis:
                        if t == 0 or ref_validity[kp_idx] > 0:
                            color = (255, 255, 255)  # White: valid/warpable
                        else:
                            color = (255, 128, 0)    # Orange: missing-in-first-frame (filled with embedding)
                        cv2.circle(canvas, (cx, cy), max(2, scale // 3), color, -1)
                        cv2.putText(canvas, str(kp_idx), (cx + 3, cy - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

                # Overlay limb connections
                all_limb_connections = _get_all_limb_connections()
                for (idx_a, idx_b) in all_limb_connections:
                    if idx_a >= kps_x.shape[1] or idx_b >= kps_x.shape[1]:
                        continue
                    if kps_v[t, idx_a] <= 0 or kps_v[t, idx_b] <= 0:
                        continue
                    ax = int(round(kps_x[t, idx_a])) * scale + scale // 2
                    ay = int(round(kps_y[t, idx_a])) * scale + scale // 2
                    bx = int(round(kps_x[t, idx_b])) * scale + scale // 2
                    by = int(round(kps_y[t, idx_b])) * scale + scale // 2
                    if idx_a < 18 and idx_b < 18:
                        line_color = (0, 255, 255)  # Cyan for body
                    else:
                        line_color = (255, 255, 0)  # Yellow for hands
                    cv2.line(canvas, (ax, ay), (bx, by), line_color, 1)

                label = f"V3 t_latent={t}/{T_latent}"
                cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                legend_y = H_vis - 45
                cv2.rectangle(canvas, (5, legend_y), (15, legend_y + 10), (0, 255, 0), -1)
                cv2.putText(canvas, "mask=1", (20, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.rectangle(canvas, (5, legend_y + 15), (15, legend_y + 25), (32, 32, 32), -1)
                cv2.putText(canvas, "mask=0", (20, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.rectangle(canvas, (5, legend_y + 30), (15, legend_y + 40), (255, 128, 0), -1)
                cv2.putText(canvas, "emb_filled", (20, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                mask_vis_frames.append(canvas)

            if mask_vis_frames:
                video_path = os.path.join(output_dir, "mask_visualization.mp4")
                imageio.mimsave(video_path, mask_vis_frames, fps=4)
                for idx in [0, 1, min(2, T_latent - 1), T_latent - 1]:
                    if idx < len(mask_vis_frames):
                        img_path = os.path.join(output_dir, f"mask_t{idx}.png")
                        imageio.imwrite(img_path, mask_vis_frames[idx])

            # Summary statistics
            import json
            num_kps = kps_x.shape[1]
            summary = {
                "version": "V3",
                "latent_shape": f"T={T_latent}, H={H_latent}, W={W_latent}",
                "num_keypoints": int(num_kps),
                "pixel_height": int(height),
                "pixel_width": int(width),
                "num_frames": int(num_frames),
                "description": "V3: same as V2 (mask 0/1, missing filled with 16ch embedding) but WITHOUT 4ch keypoint index embedding",
            }
            kp_stats = []
            for t in range(T_latent):
                valid_count = int((kps_v[t] > 0).sum())
                if t > 0:
                    both_valid = int(((kps_v[0] > 0) & (kps_v[t] > 0)).sum())
                    curr_only = int(((kps_v[0] <= 0) & (kps_v[t] > 0)).sum())
                    ref_only = int(((kps_v[0] > 0) & (kps_v[t] <= 0)).sum())
                else:
                    both_valid = valid_count
                    curr_only = 0
                    ref_only = 0
                kp_stats.append({
                    "t_latent": t,
                    "valid": valid_count,
                    "both_valid": both_valid,
                    "curr_only(emb_filled)": curr_only,
                    "ref_only(lost)": ref_only,
                })
            summary["keypoint_stats_per_frame"] = kp_stats

            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"[Sanity Check] Fix-missing warp V3 visualizations saved to {output_dir}/")

            # Warp process visualization (arrows + coverage)
            _visualize_warp_process(output_dir, kps_x, kps_y, kps_v, H_latent, W_latent, T_latent, warp_limbs_enabled=warp_limbs)

        except Exception as e:
            print(f"[Sanity Check] Error during fix_missing_warp_v3 visualization: {e}")
            import traceback
            traceback.print_exc()


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=("vap_video", "vap_prompt", "negative_vap_prompt", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap")
        )
        
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get("vap_prompt", ""), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap":vap_prompt_emb})
            inputs_nega.update({"context_vap":negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get("end_image")

            num_frames, height, width = inputs_shared.get("num_frames"),inputs_shared.get("height"), inputs_shared.get("width")
            
            image_vap = pipe.preprocess_image(vap_video[0].resize((width, height))).to(pipe.device)

            vap_clip_context = pipe.image_encoder.encode_image([image_vap])
            if end_image is not None:
                vap_end_image = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                if pipe.dit.has_image_pos_emb:
                    vap_clip_context = torch.concat([vap_clip_context, pipe.image_encoder.encode_image([vap_end_image])], dim=1)
            vap_clip_context = vap_clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_clip_feature":vap_clip_context})

            # 3. prepare vap latents            
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            if end_image is not None:
                msk[:, -1:] = 1
                last_image_vap = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                vae_input = torch.concat([image_vap.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image_vap.device), last_image_vap.transpose(0,1)],dim=1)
            else:
                vae_input = torch.concat([image_vap.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_vap.device)], dim=1)
            
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]

            tiled,tile_size,tile_stride = inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")

            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(vap_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent,y], dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_hidden_state":vap_latent})

            return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None:
            assert motion_video.shape[2] == motion_frames, f"motion video must have {motion_frames} frames, but got {motion_video.shape[2]}"
            motion_latents = motion_video
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"),
            output_params=("animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video")
        )

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            output_params=("longcat_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"longcat_latents": longcat_latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def _visualize_latent_warp_pca(image_latent_bcthw, feature_bcfhw, timestep, kwargs):
    """
    Visualize PCA of the image latent (warped) and the projected feature in the
    latent warp pipeline (LOAD_POSE3_KEY_POINTS etc.). For each call this saves
    per-frame PCA-RGB images for two groups:

      - before_proj: the (warped) image latent BEFORE the DiT patch_embedding
                     projection layer, shape [B, 16, T, H, W].
                     First frame (frame_idx=0) is the un-warped first frame;
                     other frames (frame_idx>0) are the warp output.
      - after_proj:  the DiT feature AFTER patch_embedding (Conv3d projection),
                     shape [B, C_dit, f, h, w] (C_dit=5120 for Wan2.2 A14B).

    PCA is fit jointly over all spatial tokens across all frames in the group,
    so the RGB color scheme is comparable across frames.

    Args:
        image_latent_bcthw: [B, 16, T, H, W] image latent BEFORE projection
                            (full tensor; first frame is clean, non-first frames
                            are warp output). Can be None to skip this group.
        feature_bcfhw:      [B, C_dit, f, h, w] DiT feature AFTER patch_embedding.
                            Can be None to skip this group.
        timestep: current diffusion timestep tensor (scalar-like)
        kwargs: extra keyword arguments (may contain 'latent_warp_vis_dir')
    """
    import os
    from sklearn.decomposition import PCA

    save_dir = kwargs.get('latent_warp_vis_dir', './output/latent_warp_vis')
    os.makedirs(save_dir, exist_ok=True)

    t_val = timestep.item() if hasattr(timestep, 'item') else float(timestep)

    vis_groups = {}
    if image_latent_bcthw is not None:
        vis_groups['before_proj'] = image_latent_bcthw  # [B, 16, T, H, W]
    if feature_bcfhw is not None:
        vis_groups['after_proj'] = feature_bcfhw        # [B, C_dit, f, h, w]

    with torch.no_grad():
        for group_name, group_tensor in vis_groups.items():
            if group_tensor is None:
                continue

            B, C_ch, T, H, W = group_tensor.shape

            # Use batch index 0: [C_ch, T, H, W]
            feats = group_tensor[0]

            # Rearrange to per-frame spatial tokens: [T, H*W, C_ch]
            feats = rearrange(feats, 'c t h w -> t (h w) c').contiguous()

            # Gather all spatial tokens across all frames for fitting PCA
            all_tokens = feats.reshape(T * H * W, C_ch).float().cpu().numpy()

            # Fit PCA on all tokens jointly so colors are comparable across frames
            n_components = min(3, C_ch)
            pca = PCA(n_components=n_components)
            pca.fit(all_tokens)

            group_dir = os.path.join(save_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            for frame_idx in range(T):
                tokens = feats[frame_idx].float().cpu().numpy()  # [H*W, C_ch]
                rgb = pca.transform(tokens)                      # [H*W, n_components]

                if n_components < 3:
                    padding = np.zeros((rgb.shape[0], 3 - n_components))
                    rgb = np.concatenate([rgb, padding], axis=1)

                for ch in range(3):
                    ch_min = rgb[:, ch].min()
                    ch_max = rgb[:, ch].max()
                    if ch_max - ch_min > 1e-8:
                        rgb[:, ch] = (rgb[:, ch] - ch_min) / (ch_max - ch_min) * 255.0
                    else:
                        rgb[:, ch] = 128.0

                rgb = rgb.reshape(H, W, 3).astype(np.uint8)
                img = Image.fromarray(rgb)

                tag = "first_frame" if frame_idx == 0 else f"frame_{frame_idx}"
                filename = f"t{t_val:.0f}_{tag}.png"
                img.save(os.path.join(group_dir, filename))

        print(f"[latent_warp_vis] Saved PCA visualization for timestep={t_val:.0f}, "
              f"groups={list(vis_groups.keys())} -> {save_dir}")


def _visualize_attention_warp_pca(x, timestep, kwargs, control_latent=None, image_latent=None, x_before_warp=None):
    """
    Visualize per-frame features after attention warp using PCA, operating on BCTHW format.
    Reduces feature dim to 3 (RGB) via PCA and saves per-frame images for debugging.
    
    Visualizes multiple channel groups:
      - noise:         x[:, :16]       (noisy latent)
      - image_latent:  x[:, 20:36]     (image latent after warp)
      - control:       x[:, 36:52]     (control latent)
      - raw_control:   control_latent  (original control before warp)
      - raw_image:     image_latent    (original image before warp)
      - image_before:  x_before_warp[:, 20:36] (image latent before warp, if provided)
    
    Args:
        x: tensor of shape [B, C_total, T, H, W] after attention warp
           e.g. C_total=52 = 16(noise) + 20(mask+image) + 16(control)
        timestep: current diffusion timestep tensor
        kwargs: extra keyword arguments (may contain 'attention_warp_vis_dir')
        control_latent: [B, 16, T, H, W] raw control latent (optional)
        image_latent: [B, 16, T, H, W] raw image latent (optional)
        x_before_warp: [B, C_total, T, H, W] x before attention warp (optional, for comparison)
    """
    import os
    from sklearn.decomposition import PCA

    save_dir = kwargs.get('attention_warp_vis_dir', './output/attention_warp_vis')
    os.makedirs(save_dir, exist_ok=True)

    t_val = timestep.item() if hasattr(timestep, 'item') else float(timestep)
    B, C_total, T, H, W = x.shape

    # Define channel groups to visualize
    vis_groups = {
        'noise': x[:, :16],           # [B, 16, T, H, W]
        'image_latent': x[:, 20:36],  # [B, 16, T, H, W] - after warp
        'control': x[:, 36:52] if C_total >= 52 else None,  # [B, 16, T, H, W]
    }
    if control_latent is not None:
        vis_groups['raw_control'] = control_latent  # [B, 16, T, H, W]
    if image_latent is not None:
        vis_groups['raw_image'] = image_latent      # [B, 16, T, H, W]
    if x_before_warp is not None:
        vis_groups['image_before_warp'] = x_before_warp[:, 20:36]  # [B, 16, T, H, W]

    with torch.no_grad():
        for group_name, group_tensor in vis_groups.items():
            if group_tensor is None:
                continue

            # Use batch index 0: [C_ch, T, H, W]
            feats = group_tensor[0]  # [C_ch, T, H, W]
            C_ch = feats.shape[0]

            # Rearrange to per-frame spatial tokens: [T, H*W, C_ch]
            feats = rearrange(feats, 'c t h w -> t (h w) c').contiguous()

            # Gather all spatial tokens across all frames for fitting PCA
            all_tokens = feats.reshape(T * H * W, C_ch).float().cpu().numpy()  # [T*H*W, C_ch]

            # Fit PCA on all tokens jointly so colors are comparable across frames
            n_components = min(3, C_ch)
            pca = PCA(n_components=n_components)
            pca.fit(all_tokens)

            group_dir = os.path.join(save_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            for frame_idx in range(T):
                tokens = feats[frame_idx].float().cpu().numpy()  # [H*W, C_ch]
                rgb = pca.transform(tokens)  # [H*W, n_components]

                # Pad to 3 channels if needed
                if n_components < 3:
                    padding = np.zeros((rgb.shape[0], 3 - n_components))
                    rgb = np.concatenate([rgb, padding], axis=1)

                # Normalize each channel to [0, 255]
                for ch in range(3):
                    ch_min = rgb[:, ch].min()
                    ch_max = rgb[:, ch].max()
                    if ch_max - ch_min > 1e-8:
                        rgb[:, ch] = (rgb[:, ch] - ch_min) / (ch_max - ch_min) * 255.0
                    else:
                        rgb[:, ch] = 128.0

                rgb = rgb.reshape(H, W, 3).astype(np.uint8)
                img = Image.fromarray(rgb)

                tag = "first_frame" if frame_idx == 0 else f"frame_{frame_idx}"
                filename = f"t{t_val:.0f}_{tag}.png"
                img.save(os.path.join(group_dir, filename))

        print(f"[attention_warp_vis] Saved PCA visualization for timestep={t_val:.0f}, "
              f"shape=({B},{C_total},{T},{H},{W}), groups={list(vis_groups.keys())} -> {save_dir}")

def _visualize_attention_warp_decode(x, timestep, kwargs, vae, x_before_warp=None, control_latent=None, image_latent=None, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
    """
    Visualize attention warp results by directly decoding image latent channels
    through the VAE decoder. This provides pixel-space visualization (unlike PCA
    which only shows feature-space structure).
    
    Decodes multiple latent groups per-frame and saves them as images:
      - image_after_warp:   x[:, 20:36] (image latent after warp)
      - image_before_warp:  x_before_warp[:, 20:36] (image latent before warp, if provided)
      - raw_image:          image_latent (original image latent input, if provided)
      - raw_control:        control_latent (control latent, if provided)
    
    Each group is decoded frame-by-frame through VAE and saved as PNG images.
    
    Args:
        x: tensor [B, C_total, T, H, W] after attention warp
        timestep: current diffusion timestep tensor
        kwargs: extra keyword arguments (may contain 'attention_warp_vis_dir')
        vae: WanVideoVAE instance for decoding latents to pixel space
        x_before_warp: [B, C_total, T, H, W] x before warp (optional, for comparison)
        control_latent: [B, 16, T, H, W] raw control latent (optional)
        image_latent: [B, 16, T, H, W] raw image latent (optional)
        tiled: whether to use tiled decoding
        tile_size: tile size for tiled decoding
        tile_stride: tile stride for tiled decoding
    """
    import os

    save_dir = kwargs.get('attention_warp_vis_dir', './output/attention_warp_vis')
    os.makedirs(save_dir, exist_ok=True)

    t_val = timestep.item() if hasattr(timestep, 'item') else float(timestep)
    B, C_total, T, H, W = x.shape

    # Define latent groups to decode
    decode_groups = {
        'image_after_warp': x[:, 20:36],  # [B, 16, T, H, W]
    }
    if x_before_warp is not None:
        decode_groups['image_before_warp'] = x_before_warp[:, 20:36]  # [B, 16, T, H, W]
    if image_latent is not None:
        decode_groups['raw_image'] = image_latent  # [B, 16, T, H, W]
    if control_latent is not None:
        decode_groups['raw_control'] = control_latent  # [B, 16, T, H, W]

    with torch.no_grad():
        for group_name, latent_tensor in decode_groups.items():
            if latent_tensor is None:
                continue

            group_dir = os.path.join(save_dir, f"decode_{group_name}")
            os.makedirs(group_dir, exist_ok=True)

            # Decode per-frame: extract each frame as [B, C, 1, H, W] and decode
            for frame_idx in range(T):
                frame_latent = latent_tensor[:, :, frame_idx:frame_idx+1, :, :]  # [B, 16, 1, H, W]

                try:
                    # VAE decode expects [B, C, T, H, W]; decode single frame
                    frame_video = vae.decode(
                        frame_latent,
                        device=frame_latent.device,
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )  # returns pixel-space tensor, typically [B, C, T, H_pixel, W_pixel]

                    # Convert to image: clamp to [0, 1], take first batch and first frame
                    if frame_video.dim() == 5:
                        frame_pixels = frame_video[0, :, 0]  # [C, H_pixel, W_pixel]
                    elif frame_video.dim() == 4:
                        frame_pixels = frame_video[0]  # [C, H_pixel, W_pixel]
                    else:
                        frame_pixels = frame_video

                    # Normalize to [0, 255]: VAE output is typically in [-1, 1] or [0, 1]
                    frame_pixels = frame_pixels.float().clamp(-1, 1)
                    frame_pixels = ((frame_pixels + 1) / 2 * 255).clamp(0, 255).byte()
                    frame_pixels = frame_pixels.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

                    img = Image.fromarray(frame_pixels)

                    tag = "first_frame" if frame_idx == 0 else f"frame_{frame_idx}"
                    filename = f"t{t_val:.0f}_{tag}.png"
                    img.save(os.path.join(group_dir, filename))

                except Exception as e:
                    print(f"[attention_warp_decode] Failed to decode {group_name} frame {frame_idx}: {e}")
                    continue

        print(f"[attention_warp_decode] Saved decoded visualization for timestep={t_val:.0f}, "
              f"shape=({B},{C_total},{T},{H},{W}), groups={list(decode_groups.keys())} -> {save_dir}")


# def _apply_attention_warp(x, dit, f, h, w, control_tokens, image_tokens):
#     """
#     Apply attention-based warp using control latent as Q/K and image latent as V.
#     This is called ONCE before the DiT block loop.
    
#     For each non-first frame, compute cross-attention where:
#     - Q: that frame's control latent tokens (to find correspondence)
#     - K: first frame's control latent tokens (reference structure)
#     - V: first frame's image latent tokens (reference appearance)
#     The attention output replaces the image latent of non-first frames.
    
#     Args:
#         x: hidden states of shape [B, S, C] where S >= f * h * w
#         dit: the DiT model containing attn_warp_q/k/v/o projections
#         f: number of temporal frames in latent space
#         h: spatial height in latent space
#         w: spatial width in latent space
#         control_tokens: patchified control latent tokens [B, f*h*w, C]
#         image_tokens: patchified image latent tokens [B, f*h*w, C]
    
#     Returns:
#         x: updated hidden states with warped image latent for non-first frames
#     """
#     B, S, C = x.shape
#     num_heads = dit.blocks[0].self_attn.num_heads
#     hw = h * w

#     # Check if dit has attention_warp layers
#     if not hasattr(dit, 'attn_warp_q'):
#         print("Warning: dit does not have attention_warp layers, skipping attention warp.")
#         return x

#     # Reshape control_tokens and image_tokens to per-frame: [B, f, h*w, C]
#     control_per_frame = control_tokens[:, :f * hw].reshape(B, f, hw, C)
#     image_per_frame = image_tokens[:, :f * hw].reshape(B, f, hw, C)

#     # First frame's control and image tokens as K/V source
#     first_control = control_per_frame[:, :1, :, :]  # [B, 1, hw, C]
#     first_image = image_per_frame[:, :1, :, :]      # [B, 1, hw, C]

#     # Non-first frames' control tokens as Q
#     other_control = control_per_frame[:, 1:, :, :]  # [B, f-1, hw, C]

#     num_other_frames = f - 1
#     if num_other_frames == 0:
#         return x

#     # Flatten for projection: merge frame and spatial dims
#     # Q from non-first frames' control: [B, (f-1)*hw, C]
#     q_input = other_control.reshape(B, num_other_frames * hw, C)
#     # K from first frame's control (broadcast to match): [B, hw, C]
#     k_input = first_control.reshape(B, hw, C)
#     # V from first frame's image: [B, hw, C]
#     v_input = first_image.reshape(B, hw, C)

#     # Compute Q/K/V through dedicated projections on dit
#     q = dit.attn_warp_norm_q(dit.attn_warp_q(q_input))  # [B, (f-1)*hw, C]
#     k = dit.attn_warp_norm_q(dit.attn_warp_q(k_input))  # [B, hw, C]
#     v = dit.attn_warp_v(v_input)                          # [B, hw, C]

#     # Per-frame attention: reshape Q to [B*(f-1), hw, C], broadcast K/V to [B*(f-1), hw, C]
#     q = q.reshape(B * num_other_frames, hw, C)
#     k = k.unsqueeze(1).expand(B, num_other_frames, hw, C).reshape(B * num_other_frames, hw, C)
#     v = v.unsqueeze(1).expand(B, num_other_frames, hw, C).reshape(B * num_other_frames, hw, C)

#     # Flash attention: [B*(f-1), hw, C]
#     attn_out = flash_attention(q, k, v, num_heads=num_heads)

#     # Project through output
#     attn_out = attn_out.reshape(B, num_other_frames * hw, C)
#     attn_out = dit.attn_warp_o(attn_out)

#     # Use the attention warp result to replace non-first frame tokens
#     x[:, hw:f * hw, :] = attn_out

#     return x

# v9
# def _apply_attention_warp(x, dit, control_latent, image_latent):
#     """
#     Apply attention-based warp using control latent as Q/K and image latent as V.
#     This is called ONCE before the DiT block loop, operating in BCTHW format.
    
#     For each non-first frame, compute cross-attention where:
#     - Q: that frame's control latent (to find correspondence)
#     - K: first frame's control latent (reference structure)
#     - V: first frame's image latent (reference appearance)
#     The attention output is written back into the image latent channels of x
#     for non-first frames.
    
#     Args:
#         x: tensor of shape [B, C_total, T, H, W] (before patchify),
#            e.g. C_total=52 = 16(noise) + 20(mask+image) + 16(control)
#         dit: the DiT model containing attn_warp_q/k/v/o projections
#         control_latent: [B, C_lat, T, H, W] raw control latent (C_lat=16)
#         image_latent: [B, C_lat, T, H, W] raw image latent (C_lat=16)
    
#     Returns:
#         x: updated tensor in [B, C_total, T, H, W] with warped image latent
#            for non-first frames
#     """
#     B, C_total, T, H, W = x.shape
#     C_proj = control_latent.shape[1]  # typically 16
#     C_lat = 16  # typically 16
#     hw = H * W

#     # Check if dit has attention_warp layers
#     if not hasattr(dit, 'attn_warp_q'):
#         raise Exception("dit does not have attention_warp layers, cannot apply attention warp.")

#     # num_heads must satisfy: C_lat % num_heads == 0 and head_dim = C_lat // num_heads
#     # must be supported by FlashAttention (min head_dim ~ 8 for FA2).
#     # With C_lat=16, num_heads=1 gives head_dim=16 (safe for all FA versions).
#     num_heads = dit.blocks[0].self_attn.num_heads

#     # Reshape BCTHW to per-frame spatial tokens: [B, T, H*W, C_lat]
#     control_per_frame = rearrange(control_latent, 'b c t h w -> b t (h w) c').contiguous()
#     image_per_frame = rearrange(image_latent, 'b c t h w -> b t (h w) c').contiguous()

#     # First frame tokens as K/V source
#     first_control = control_per_frame[:, :1, :, :]  # [B, 1, hw, C_lat]
#     first_image = image_per_frame[:, :1, :, :]      # [B, 1, hw, C_lat]

#     # # Non-first frames' control tokens as Q
#     # other_control = control_per_frame[:, 1:, :, :]  # [B, T-1, hw, C_lat]
#     # num_other_frames = T - 1

#     # All frames' control tokens as Q
#     other_control = control_per_frame[:, :, :, :]  # [B, T, hw, C_lat]
#     num_other_frames = T

#     if num_other_frames == 0:
#         return x

#     # Flatten for projection: merge frame and spatial dims
#     q_input = other_control.reshape(B, num_other_frames * hw, C_proj)  # [B, T*hw, C_proj]
#     k_input = first_control.reshape(B, hw, C_proj)                     # [B, hw, C_proj]
#     v_input = first_image.reshape(B, hw, C_proj)                       # [B, hw, C_proj]

#     # Compute Q/K through dedicated projections on dit
#     q = dit.attn_warp_norm_q(dit.attn_warp_q(q_input))  # [B, T*hw, C_proj]
#     k = dit.attn_warp_norm_k(dit.attn_warp_k(k_input))  # [B, hw, C_proj]
#     v = dit.attn_warp_norm_v(dit.attn_warp_v(v_input))  # [B, hw, C_proj]

#     C_proj = q.shape[-1]

#     # Per-frame attention: reshape Q to [B*(T-1), hw, C_proj], broadcast K/V
#     q = q.reshape(B * num_other_frames, hw, C_proj)
#     k = k.unsqueeze(1).expand(B, num_other_frames, hw, C_proj).reshape(B * num_other_frames, hw, C_proj)
#     v = v.unsqueeze(1).expand(B, num_other_frames, hw, C_proj).reshape(B * num_other_frames, hw, C_proj)

#     # Flash attention: [B*(T-1), hw, C_proj]
#     attn_out = flash_attention(q, k, v, num_heads=num_heads)

#     # Reshape back to [B, T-1, H, W, C_proj] then to BCTHW
#     attn_out = attn_out.reshape(B, num_other_frames * hw, C_proj)
#     # print(f"[apply_attention_warp] attn_out.shape={attn_out.shape}")
#     attn_out = dit.attn_warp_o(attn_out)
#     attn_out = dit.attn_warp_result_proj(attn_out)
#     # print(f"[apply_attention_warp] attn_out.shape={attn_out.shape}")
    
#     attn_out = attn_out.reshape(B, num_other_frames, H, W, C_lat)
#     attn_out = rearrange(attn_out, 'b t h w c -> b c t h w').contiguous()

#     # Write warped image latent back into x for non-first frames
#     # Image latent occupies channels 20:36 in the 52-channel x (after noise(16) + mask(4))
#     x = x.clone()

#     # x[:, 20:36, 1:, :, :] = attn_out
#     x[:, 20:36, :, :, :] = attn_out

#     return x

# v10
def _apply_attention_warp(x, dit, control_latent, image_latent):
    """
    Apply attention-based warp using control latent as Q/K and image latent as V.
    This is called ONCE before the DiT block loop, operating in BCTHW format.
    
    For each non-first frame, compute cross-attention where:
    - Q: that frame's control latent (to find correspondence)
    - K: first frame's control latent (reference structure)
    - V: first frame's image latent (reference appearance)
    The attention output is written back into the image latent channels of x
    for non-first frames.
    
    Args:
        x: tensor of shape [B, C_total, T, H, W] (before patchify),
           e.g. C_total=52 = 16(noise) + 20(mask+image) + 16(control)
        dit: the DiT model containing attn_warp_q/k/v/o projections
        control_latent: [B, C_lat, T, H, W] raw control latent (C_lat=16)
        image_latent: [B, C_lat, T, H, W] raw image latent (C_lat=16)
    
    Returns:
        x: updated tensor in [B, C_total, T, H, W] with warped image latent
           for non-first frames
    """
    B, C_total, T, H, W = x.shape
    C_proj = control_latent.shape[1]  # typically 16
    C_lat = 16  # typically 16
    hw = H * W

    # Check if dit has attention_warp layers
    if not hasattr(dit, 'attn_warp_q'):
        raise Exception("dit does not have attention_warp layers, cannot apply attention warp.")

    # num_heads must satisfy: C_lat % num_heads == 0 and head_dim = C_lat // num_heads
    # must be supported by FlashAttention (min head_dim ~ 8 for FA2).
    # With C_lat=16, num_heads=1 gives head_dim=16 (safe for all FA versions).
    # num_heads = dit.blocks[0].self_attn.num_heads
    num_heads = 1

    # Reshape BCTHW to per-frame spatial tokens: [B, T, H*W, C_lat]
    control_per_frame = rearrange(control_latent, 'b c t h w -> b t (h w) c').contiguous()
    image_per_frame = rearrange(image_latent, 'b c t h w -> b t (h w) c').contiguous()

    # First frame tokens as K/V source
    first_control = control_per_frame[:, :1, :, :]  # [B, 1, hw, C_lat]
    first_image = image_per_frame[:, :1, :, :]      # [B, 1, hw, C_lat]

    # # Non-first frames' control tokens as Q
    # other_control = control_per_frame[:, 1:, :, :]  # [B, T-1, hw, C_lat]
    # num_other_frames = T - 1

    # All frames' control tokens as Q
    other_control = control_per_frame[:, :, :, :]  # [B, T, hw, C_lat]
    num_other_frames = T

    if num_other_frames == 0:
        return x

    # Flatten for projection: merge frame and spatial dims
    q_input = other_control.reshape(B, num_other_frames * hw, C_proj)  # [B, T*hw, C_proj]
    k_input = first_control.reshape(B, hw, C_proj)                     # [B, hw, C_proj]
    v_input = first_image.reshape(B, hw, C_proj)                       # [B, hw, C_proj]

    # Compute Q/K through dedicated projections on dit
    q = dit.attn_warp_norm_q(dit.attn_warp_q(q_input))  # [B, T*hw, C_proj]
    k = dit.attn_warp_norm_k(dit.attn_warp_k(k_input))  # [B, hw, C_proj]
    # v = dit.attn_warp_norm_v(dit.attn_warp_v(v_input))  # [B, hw, C_proj]
    v = v_input  # [B, hw, C_proj]

    # C_proj = q.shape[-1]

    # Per-frame attention: reshape Q to [B*(T-1), hw, C_proj], broadcast K/V
    q = q.reshape(B * num_other_frames, hw, C_proj)
    k = k.unsqueeze(1).expand(B, num_other_frames, hw, C_proj).reshape(B * num_other_frames, hw, C_proj)
    v = v.unsqueeze(1).expand(B, num_other_frames, hw, C_proj).reshape(B * num_other_frames, hw, C_proj)

    # Flash attention: [B*(T-1), hw, C_proj]
    # attn_out = flash_attention(q, k, v, num_heads=num_heads)

    # force to use fa2
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    attn_out = F.scaled_dot_product_attention(q, k, v)
    attn_out = rearrange(attn_out, "b n s d -> b s (n d)", n=num_heads)


    # Reshape back to [B, T-1, H, W, C_proj] then to BCTHW
    attn_out = attn_out.reshape(B, num_other_frames * hw, C_proj)
    # print(f"[apply_attention_warp] attn_out.shape={attn_out.shape}")
    # attn_out = dit.attn_warp_o(attn_out)
    # attn_out = dit.attn_warp_result_proj(attn_out)
    # print(f"[apply_attention_warp] attn_out.shape={attn_out.shape}")
    
    attn_out = attn_out.reshape(B, num_other_frames, H, W, C_lat)
    attn_out = rearrange(attn_out, 'b t h w c -> b c t h w').contiguous()

    # Write warped image latent back into x for non-first frames
    # Image latent occupies channels 20:36 in the 52-channel x (after noise(16) + mask(4))
    x = x.clone()

    # x[:, 20:36, 1:, :, :] = attn_out
    x[:, 20:36, :, :, :] = attn_out

    return x


def _apply_ref_detail_transfer(x, block, f, h, w, ref_hw, adaptive_scale, ref_latent_tokens=None):
    """
    Apply reference frame detail transfer via cross-attention.
    
    Uses the self-attention Q/K/V projections from the DiT block to compute
    cross-attention between non-reference frames (as queries) and the lossless
    reference frame latent tokens (as keys/values), transferring high-frequency
    detail and texture from the original VAE-encoded reference frame.
    
    In the first_as_guidance pipeline (no reference_latents), the token layout
    of x is: [first_frame (h*w), other_frames ((f-1)*h*w), optional_control_tokens].
    Only the other_frames portion is modified; first_frame and any trailing
    control tokens remain untouched.
    
    Args:
        x: hidden states of shape [B, S, C] where S >= f * h * w
        block: the current DiTBlock containing self_attn with Q/K/V projections
        f: number of temporal frames in latent space
        h: spatial height in latent space
        w: spatial width in latent space  
        ref_hw: number of spatial tokens per frame (h * w)
        adaptive_scale: timestep-adaptive blending strength
        ref_latent_tokens: lossless reference frame tokens from VAE encoding
                           (after patch_embedding), shape [B, h*w, C]. If None,
                           falls back to extracting from x (not recommended).
    
    Returns:
        x: updated hidden states with reference frame details transferred
    """
    B, S, C = x.shape
    num_heads = block.self_attn.num_heads
    
    # Compute the token range for the main latent (excluding any trailing control tokens)
    main_tokens_end = f * h * w   # total main latent tokens
    
    # Use lossless reference latent tokens for K/V source
    if ref_latent_tokens is not None:
        ref_tokens = ref_latent_tokens       # [B, h*w, C] - lossless VAE-encoded reference
    else:
        ref_tokens = x[:, :ref_hw, :]        # [B, h*w, C] - fallback: smoothed reference from x
    
    # Only operate on non-first-frame tokens within the main latent range
    # other_tokens = x[:, ref_hw:main_tokens_end, :]   # [B, (f-1)*h*w, C] - other frames only
    other_tokens = x[:, :main_tokens_end, :]   # [B, (f)*h*w, C] - other frames only
    
    if other_tokens.shape[1] == 0:
        return x
    
    # Use dedicated ref_detail Q/K/V/O layers if available (zero-initialized, trainable),
    # otherwise fall back to self-attention projections (training-free).
    self_attn = block.self_attn

    ### check if ref_detail_q in block and ref_detail_k in block and ref_detail_v in block and ref_detail_o in block
    # print(f"block: {block}")
    has_ref_detail = all(hasattr(block, f'ref_detail_{x}') for x in ['q', 'k', 'v', 'o'])
    if not has_ref_detail:
        print(f"warning: has_ref_detail: {has_ref_detail}")

    ref_q = getattr(block, 'ref_detail_q', None) or self_attn.q
    ref_k = getattr(block, 'ref_detail_k', None) or self_attn.k
    ref_v = getattr(block, 'ref_detail_v', None) or self_attn.v
    ref_o = getattr(block, 'ref_detail_o', None) or self_attn.o
    
    # Compute Q from non-reference frames, K/V from lossless reference frame
    q = self_attn.norm_q(ref_q(other_tokens))    # [B, (f-1)*h*w, C]
    k = self_attn.norm_k(ref_k(ref_tokens))      # [B, h*w, C]
    v = ref_v(ref_tokens)                         # [B, h*w, C]
    
    # Use flash_attention which auto-selects the best backend:
    # Flash Attn 3 > Flash Attn 2 > SageAttn > SDPA fallback
    ref_attn_out = flash_attention(q, k, v, num_heads=num_heads)  # [B, (f-1)*h*w, C]
    
    # Project through output projection
    ref_attn_out = ref_o(ref_attn_out)
    
    # Blend: add reference detail to non-first-frame tokens only (leave control tokens untouched)
    x = x.clone()
    
    # print(x.shape, other_tokens.shape, ref_attn_out.shape);assert 0 # torch.Size([1, 23400, 5120]) torch.Size([1, 23400, 5120]) torch.Size([1, 23400, 5120])

    x[:, :main_tokens_end, :] = other_tokens + adaptive_scale * ref_attn_out
    
    return x


def _apply_ref_detail_transfer_adaptive(x, block, f, h, w, ref_hw, adaptive_scale, ref_latent_tokens=None):
    """
    Apply reference frame detail transfer via cross-attention with per-block learnable adaptive weight.
    
    Same as _apply_ref_detail_transfer but uses block.ref_detail_adaptive_weight (a learnable
    scalar parameter) to modulate the blending strength, enabling the model to learn how much
    reference detail to transfer at each block independently.
    
    Args:
        x: hidden states of shape [B, S, C] where S >= f * h * w
        block: the current DiTBlock containing self_attn with Q/K/V projections
               and ref_detail_adaptive_weight (learnable scalar)
        f: number of temporal frames in latent space
        h: spatial height in latent space
        w: spatial width in latent space  
        ref_hw: number of spatial tokens per frame (h * w)
        adaptive_scale: timestep-adaptive blending strength
        ref_latent_tokens: lossless reference frame tokens from VAE encoding
    
    Returns:
        x: updated hidden states with reference frame details transferred
    """
    B, S, C = x.shape
    num_heads = block.self_attn.num_heads
    
    main_tokens_end = f * h * w
    
    if ref_latent_tokens is not None:
        ref_tokens = ref_latent_tokens
    else:
        ref_tokens = x[:, :ref_hw, :]
    
    other_tokens = x[:, :main_tokens_end, :]
    
    if other_tokens.shape[1] == 0:
        return x
    
    self_attn = block.self_attn
    has_ref_detail = all(hasattr(block, f'ref_detail_{n}') for n in ['q', 'k', 'v', 'o'])
    if not has_ref_detail:
        print(f"warning: has_ref_detail: {has_ref_detail}")

    ref_q = getattr(block, 'ref_detail_q', None) or self_attn.q
    ref_k = getattr(block, 'ref_detail_k', None) or self_attn.k
    ref_v = getattr(block, 'ref_detail_v', None) or self_attn.v
    ref_o = getattr(block, 'ref_detail_o', None) or self_attn.o
    
    q = self_attn.norm_q(ref_q(other_tokens))
    k = self_attn.norm_k(ref_k(ref_tokens))
    v = ref_v(ref_tokens)
    
    ref_attn_out = flash_attention(q, k, v, num_heads=num_heads)
    ref_attn_out = ref_o(ref_attn_out)
    
    # Apply learnable per-block adaptive weight (sigmoid to keep in [0, 1])
    learned_weight = torch.sigmoid(block.ref_detail_adaptive_weight)
    
    x = x.clone()
    x[:, :main_tokens_end, :] = other_tokens + adaptive_scale * learned_weight * ref_attn_out
    
    return x


def _prepare_cross_attn_concat(x, ref_latent_tokens, freqs, ref_hw):
    """
    Prepare concatenated sequence for first_as_guidance_cross_attn.
    
    Prepends first-frame clean latent tokens to the token sequence so that
    all tokens can attend to the reference frame through standard self-attention.
    Also extends RoPE frequencies accordingly.
    
    Args:
        x: hidden states of shape [B, S, C]
        ref_latent_tokens: reference frame tokens [B, h*w, C]
        freqs: RoPE frequencies [N, 1, D]
        ref_hw: number of spatial tokens per frame (h * w)
    
    Returns:
        x_concat: concatenated sequence [B, h*w + S, C]
        freqs_concat: extended RoPE frequencies [h*w + N, 1, D]
        num_ref_tokens: number of prepended reference tokens
    """
    num_ref_tokens = ref_latent_tokens.shape[1]  # h * w
    
    # Prepend reference tokens: [ref_tokens, x] -> [B, num_ref_tokens + S, C]
    x_concat = torch.cat([ref_latent_tokens, x], dim=1)
    
    # Extend freqs: use first-frame RoPE positions for the prepended reference tokens
    ref_freqs = freqs[:ref_hw]  # [h*w, 1, D]
    freqs_concat = torch.cat([ref_freqs, freqs], dim=0)
    
    return x_concat, freqs_concat, num_ref_tokens


def _remove_cross_attn_concat(x, num_ref_tokens):
    """
    Remove the prepended reference tokens after block forward pass.
    
    Args:
        x: hidden states with prepended ref tokens [B, num_ref_tokens + S, C]
        num_ref_tokens: number of prepended reference tokens to remove
    
    Returns:
        x: hidden states without ref tokens [B, S, C]
    """
    return x[:, num_ref_tokens:, :]


def _build_depth_aware_freqs(dit, f, h, w, depth_keypoints, key_points, patch_size, device,
                              depth_levels=64, default_depth_level=32):
    """Build depth-aware 4D RoPE frequencies using keypoint depth information.
    
    Converts sparse per-keypoint depth values into a dense per-patch depth map,
    then constructs 4D RoPE frequencies (frame, height, width, depth).
    
    Args:
        dit: WanModel with freqs_with_depth attribute (f_freqs, h_freqs, w_freqs, d_freqs)
        f, h, w: patch grid dimensions (after patchify)
        depth_keypoints: tensor [T_pixel, N, 2] where [:,:,0]=depth (0~1), [:,:,1]=validity
        key_points: tensor [T_pixel, N, 3] where [:,:,0]=x_norm, [:,:,1]=y_norm, [:,:,2]=validity
        patch_size: tuple (pt, ph, pw) - temporal and spatial patch sizes
        device: target device
        depth_levels: number of discrete depth levels for quantization
        default_depth_level: default depth index for patches without keypoint coverage
    
    Returns:
        freqs: tensor [f*h*w, 1, D] complex RoPE frequencies with depth dimension
    """
    pt, ph, pw = patch_size
    T_pixel = depth_keypoints.shape[0]
    N = depth_keypoints.shape[1]
    
    # Ensure tensors are on the same device
    if isinstance(depth_keypoints, torch.Tensor):
        dk = depth_keypoints.float().cpu()
    else:
        dk = torch.tensor(depth_keypoints, dtype=torch.float32)
    
    if isinstance(key_points, torch.Tensor):
        kp = key_points.float().cpu()
    else:
        kp = torch.tensor(key_points, dtype=torch.float32)
    
    # Build dense depth map at patch resolution [f, h, w]
    # Each patch covers pt frames temporally, ph pixels in height, pw pixels in width
    # Pixel resolution: T_pixel frames, H_pixel = h * ph, W_pixel = w * pw (approximately)
    depth_map = torch.zeros((f, h, w), dtype=torch.float32)
    depth_count = torch.zeros((f, h, w), dtype=torch.float32)

    # Vectorized scatter: avoid Python double loop for performance
    T_used = min(T_pixel, f * pt)
    # Flatten over (time, keypoint) dimensions
    dk_flat = dk[:T_used].reshape(-1, 2)       # [T_used*N, 2]
    kp_flat = kp[:T_used].reshape(-1, 3)       # [T_used*N, 3]

    # Build frame indices for each pixel frame, repeated N times
    t_pixel_idx = torch.arange(T_used).unsqueeze(1).expand(T_used, N).reshape(-1)  # [T_used*N]
    t_patch_idx = t_pixel_idx // pt  # [T_used*N]

    # Validity mask: both keypoint and depth must be valid, and patch in range
    valid = (kp_flat[:, 2] >= 0.5) & (dk_flat[:, 1] >= 0.5) & (t_patch_idx < f)

    # Apply validity mask
    t_patch_valid = t_patch_idx[valid]
    x_norm_valid = kp_flat[valid, 0]
    y_norm_valid = kp_flat[valid, 1]
    depth_valid = dk_flat[valid, 0]

    # Convert normalized coords to patch indices, clamped to valid range
    h_patch_valid = (y_norm_valid * h).long().clamp(0, h - 1)
    w_patch_valid = (x_norm_valid * w).long().clamp(0, w - 1)

    # Scatter-add depth values and counts
    flat_idx = t_patch_valid * (h * w) + h_patch_valid * w + w_patch_valid
    depth_map.view(-1).scatter_add_(0, flat_idx, depth_valid * (depth_levels - 1))
    depth_count.view(-1).scatter_add_(0, flat_idx, torch.ones_like(depth_valid))

    # Average where we have keypoint coverage; use default for uncovered patches
    has_coverage = depth_count > 0
    depth_map[has_coverage] = depth_map[has_coverage] / depth_count[has_coverage]
    depth_map[~has_coverage] = float(default_depth_level)
    
    # Quantize to integer depth indices
    depth_indices = depth_map.long().clamp(0, depth_levels - 1)  # [f, h, w]
    
    # Build 4D freqs using freqs_with_depth
    f_freqs, h_freqs, w_freqs, d_freqs = dit.freqs_with_depth
    
    freqs = torch.cat([
        f_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        d_freqs[depth_indices],  # [f, h, w, d_dim//2] - per-patch depth encoding
    ], dim=-1).reshape(f * h * w, 1, -1)
    
    return freqs


def _build_depth_aware_freqs_v2(dit, f, h, w, depth_indices, patch_size, device,
                                 depth_levels=64, default_depth_level=32):
    """Build depth-aware 4D RoPE frequencies using pre-computed depth indices.
    
    Accepts pre-quantized depth_indices [f, h, w] int64 tensor (computed in DataLoader
    by LoadDepthKeypoints2), avoiding expensive GPU->CPU synchronization and redundant
    downsampling/quantization that previously happened every forward pass.
    
    Args:
        dit: WanModel with freqs_with_depth attribute (f_freqs, h_freqs, w_freqs, d_freqs)
        f, h, w: patch grid dimensions (after patchify)
        depth_indices: int64 tensor [f, h, w] with values in [0, depth_levels-1],
                      pre-computed by LoadDepthKeypoints2 in the DataLoader.
        patch_size: tuple (pt, ph, pw) - temporal and spatial patch sizes (unused, kept for API compat)
        device: target device
        depth_levels: number of discrete depth levels for quantization
        default_depth_level: default depth index for patches without depth coverage
    
    Returns:
        freqs: tensor [f*h*w, 1, D] complex RoPE frequencies with depth dimension
    """
    # depth_indices is already quantized int64 [f, h, w] from DataLoader
    # Move to CPU since freqs_with_depth is pre-computed on CPU
    if isinstance(depth_indices, torch.Tensor):
        di = depth_indices.long().cpu()
    else:
        di = torch.tensor(depth_indices, dtype=torch.long)
    
    # Handle shape mismatch gracefully (e.g., if DataLoader grid != model grid)
    if di.shape != (f, h, w):
        di = torch.nn.functional.interpolate(
            di.float().unsqueeze(0).unsqueeze(0), size=(f, h, w), mode='nearest'
        ).squeeze(0).squeeze(0).long()
    
    di = di.clamp(0, depth_levels - 1)
    
    # Build 4D freqs using freqs_with_depth
    f_freqs, h_freqs, w_freqs, d_freqs = dit.freqs_with_depth
    
    freqs = torch.cat([
        f_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        d_freqs[di],  # [f, h, w, d_dim//2] - per-patch depth encoding
    ], dim=-1).reshape(f * h * w, 1, -1)
    
    return freqs


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    # print(x.shape, y.shape); assert 0 # torch.Size([1, 16, 13, 104, 72]) torch.Size([1, 36, 13, 104, 72]) ip2v pipeline
    # print(x.shape, y.shape); assert 0 # torch.Size([1, 16, 13, 104, 72]) torch.Size([1, 20, 13, 104, 72]) latent warp pipeline
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Attention warp: extract raw control/image latent BEFORE patchify (x still has 52 channels)
    attn_warp_control_raw = None
    attn_warp_image_raw = None
    if kwargs.get('attention_warp'):
        # print(x.shape);assert 0 # torch.Size([1, 52, 21, 104, 72]) 
        # x layout: x_original(16) + image_latent(20: 4 mask + 16 image) + control_latent(16) = 52 channels
        attn_warp_control_raw = x[:, -16:, :, :, :].clone()   # [B, 16, T, H, W]
        attn_warp_image_raw = x[:, 20:36, :, :, :].clone()    # [B, 16, T, H, W]
        # print(f"attn_warp_control_raw: {attn_warp_control_raw.shape}", f"attn_warp_image_raw: {attn_warp_image_raw.shape}");assert 0
        # attn_warp_control_raw: torch.Size([1, 16, 21, 104, 72]) attn_warp_image_raw: torch.Size([1, 16, 21, 104, 72])


        # Patchify control_latent and image_latent through dedicated embeddings
        attn_warp_control_tokens = dit.attn_warp_control_patchify(attn_warp_control_raw)  # [B, C, f, h', w']
        attn_warp_image_tokens = attn_warp_image_raw
        # attn_warp_image_tokens = dit.attn_warp_image_patchify(attn_warp_image_raw)  # [B, C, f, h', w']
        # print(f"attn_warp_control_tokens: {attn_warp_control_tokens.shape}", f"attn_warp_image_tokens: {attn_warp_image_tokens.shape}")
        # attn_warp_control_tokens: torch.Size([1, 5120, 21, 60, 104]) attn_warp_image_tokens: torch.Size([1, 5120, 21, 60, 104])

        # Attention warp: apply once before patchify, using BCTHW format
        x_before_warp = x.clone() if kwargs.get('attention_warp_vis') else None
        x = _apply_attention_warp(x, dit, attn_warp_control_tokens, attn_warp_image_tokens)
        x = x[:, :36, :, :, :] # remove pose tokens in x

        # PCA visualization of attention warp results for debugging
        if kwargs.get('attention_warp_vis'):
            _visualize_attention_warp_pca(
                x, timestep, kwargs,
                control_latent=attn_warp_control_raw,
                image_latent=attn_warp_image_raw,
                x_before_warp=x_before_warp,
            )

        # Decode visualization: directly decode image latent to pixel space via VAE
        if kwargs.get('attention_warp_vis') and kwargs.get('attention_warp_vae') is not None:
            _visualize_attention_warp_decode(
                x, timestep, kwargs,
                vae=kwargs['attention_warp_vae'],
                x_before_warp=x_before_warp,
                control_latent=attn_warp_control_raw,
                image_latent=attn_warp_image_raw,
                tiled=kwargs.get('tiled', True),
                tile_size=kwargs.get('tile_size', (30, 52)),
                tile_stride=kwargs.get('tile_stride', (15, 26)),
            )


    # Camera control
    # print(x.shape, y.shape, dit.require_vae_embedding)
    # print(kwargs.get('temporal_concat')); assert 0 # True
    if kwargs.get('temporal_concat'):
        # x is already concat with y: x_original(16) + image_latent(20) + control_latent(16) = 52 channels
        # We need to split out control_latent (last 16 channels) from the already-concat x
        assert x.shape[1] == 52, "Expected 52 channels: 16 + 36 = 52"
        # Split: x_with_image (first 36 channels) and control_latent (last 16 channels)
        x_with_image = x[:, :36, :, :, :]  # x_original(16) + image_latent(20)

        # control_latent = x[:, 36:, :, :, :]  # control_latent(16)
        control_latent = kwargs.get('y_temporal_control_latents')
        # print(x_with_image.shape, control_latent.shape); assert 0 # torch.Size([1, 36, 15, 60, 104]) torch.Size([1, 16, 15, 30, 52])
        assert control_latent is not None

        # Patchify x_with_image as usual
        x = dit.patchify(x_with_image, control_camera_latents_input)

        def apply_i2v_ones_masks(inputs: torch.Tensor, mask_dim: int = 4):
            b, d, t, h, w= inputs.shape
            mask = torch.ones(b, mask_dim, t, h, w, device=inputs.device, dtype=inputs.dtype)
            inputs = torch.concat([mask, inputs], dim=1)
            return inputs

        control_latent = apply_i2v_ones_masks(control_latent)

        # Process control_latent with patch_embedding_pose
        control_latent = dit.patchify_pose(control_latent)

        # print(x.shape, control_latent.shape);assert 0 # torch.Size([1, 5120, 21, 52, 30]) torch.Size([1, 5120, 21, 26, 15])
    else:
        # Latent warp PCA visualization: capture image latent BEFORE projection (dit.patchify).
        # For the latent warp pipeline (LOAD_POSE3_KEY_POINTS etc.), y carries the warped
        # image latent at y[:, 4:20] (mask 4ch + image_latent 16ch, regardless of fix_missing_warp
        # v1/v2/v3). Frame 0 of this tensor is the clean first-frame latent; frames 1..T-1 are
        # the warp output for non-first frames.
        latent_warp_vis_image_latent = None
        if kwargs.get('latent_warp_vis') and y is not None:
            latent_warp_vis_image_latent = y[:, 4:20, :, :, :].clone()

        # TOKEN_REPLACE: when enabled, route through the trainable clone
        # `patch_embedding_token_replace` (same shape as `patch_embedding`, weight-initialized
        # from it in train.py). Only this clone is updated during training; all other DiT
        # parameters stay frozen.
        x = dit.patchify(x, control_camera_latents_input, token_replace=bool(kwargs.get('token_replace', False)))

        # Latent warp PCA visualization: compute projected feature for the "image latent only"
        # path so the before/after pair is directly comparable. We reconstruct a minimal input
        # for dit.patch_embedding consisting of:
        #   zero x_original(16) + mask(4) + image_latent(16) = 36 channels
        # then project via dit.patch_embedding to get the after-projection feature
        # [B, C_dit, f, h, w]. This mirrors what dit.patchify does on the fused x tensor,
        # but isolated to the image-latent portion for clearer PCA visualization.
        if kwargs.get('latent_warp_vis') and latent_warp_vis_image_latent is not None:
            with torch.no_grad():
                B_v, C16_v, T_v, H_v, W_v = latent_warp_vis_image_latent.shape
                zero_x = torch.zeros(B_v, 16, T_v, H_v, W_v,
                                     device=latent_warp_vis_image_latent.device,
                                     dtype=latent_warp_vis_image_latent.dtype)
                # Build 36ch input: zero x_original(16) + y_original(20: mask 4 + image 16)
                proj_input = torch.cat([zero_x, y[:, :20, :, :, :]], dim=1)
                # If dit.patch_embedding has been expanded (+4ch) for fix_missing_warp v1/v2,
                # pad with zeros on the channel dim to match in_channels.
                pe_in = dit.patch_embedding.in_channels
                if proj_input.shape[1] < pe_in:
                    pad_c = pe_in - proj_input.shape[1]
                    pad = torch.zeros(B_v, pad_c, T_v, H_v, W_v,
                                      device=proj_input.device, dtype=proj_input.dtype)
                    proj_input = torch.cat([proj_input, pad], dim=1)
                elif proj_input.shape[1] > pe_in:
                    proj_input = proj_input[:, :pe_in]
                feat_after_proj = dit.patch_embedding(proj_input)  # [B, C_dit, f, h, w]

            _visualize_latent_warp_pca(
                image_latent_bcthw=latent_warp_vis_image_latent,
                feature_bcfhw=feat_after_proj,
                timestep=timestep,
                kwargs=kwargs,
            )
    # print(x.shape);assert 0
    

    # Reference frame detail transfer setup (basic flag, must be before first_as_guidance block)
    ref_detail_transfer_scale = kwargs.get('ref_detail_transfer_scale', 1)
    ref_detail_transfer_start = kwargs.get('ref_detail_transfer_start', 1.0)
    ref_detail_transfer_end = kwargs.get('ref_detail_transfer_end', 0.0)
    ref_detail_transfer_layers = kwargs.get('ref_detail_transfer_layers', None)
    # print(kwargs.get('first_as_guidance', False), ref_detail_transfer_scale, ref_detail_transfer_start, ref_detail_transfer_end, ref_detail_transfer_layers)
    enable_ref_detail_transfer = (
        ref_detail_transfer_scale is not None
        and ref_detail_transfer_scale > 0
        and (kwargs.get('first_as_guidance', False) or kwargs.get('first_as_guidance_middle', False) or kwargs.get('first_as_guidance_adaptive', False) or kwargs.get('first_as_guidance_cross_attn', False))
    )

    # first as guidance for latent warp pipeline
    if kwargs.get('first_as_guidance') or kwargs.get('first_as_guidance_middle') or kwargs.get('first_as_guidance_adaptive') or kwargs.get('first_as_guidance_cross_attn'):
        # Extract image latent from y (channels 4:20, always 16ch regardless of fix_missing_warp)
        first_frame_latent = y[:, 4:20, :1, :, :]
        # print(first_frame_latent.shape, x.shape);assert 0 # torch.Size([1, 16, 1, 104, 60]) torch.Size([1, 5120, 21, 52, 30])
        
        # Build lossless reference tokens for detail transfer (training-free)
        # Construct a clean first-frame input in the same format as patchify input:
        #   first_frame_latent (16ch) + y[:, :, :1] (mask 4ch + image_latent 16ch = 20ch) = 36ch
        if enable_ref_detail_transfer:
            clean_first_frame_input = torch.cat([first_frame_latent, y[:, :, :1, :, :]], dim=1)  # [B, 36, 1, H, W]
            ref_latent_tokens_for_detail = dit.patch_embedding(clean_first_frame_input)  # [B, C, 1, h, w]
            ref_latent_tokens_for_detail = rearrange(ref_latent_tokens_for_detail, 'b c f h w -> b (f h w) c').contiguous()

    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)


    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    # Depth embedding injection (additive, preserves 3D RoPE compatibility)
    # Must be applied BEFORE reference_latents concat (only video tokens get depth info)
    depth_keypoints2 = kwargs.get('depth_keypoints2', None)
    if depth_keypoints2 is not None and getattr(dit, 'depth_embedding', None) is not None:
        x = dit._apply_depth_embedding(x, f, h, w, depth_keypoints2)
    
    ### concat tokens for temporal concat
    if kwargs.get('temporal_concat'):
        control_latent = rearrange(control_latent, 'b c f h w -> b (f h w) c').contiguous()
        x = torch.concat([x, control_latent], dim=1)

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    # Always use standard 3D RoPE (depth info is injected via additive embedding above)
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1)

    ### concat freqs for temporal concat
    if kwargs.get('temporal_concat'):
        pose_rope_shift = getattr(dit, 'pose_rope_shift', [0, 0, 120])
        global_rope_H = pose_rope_shift[1]
        global_rope_W = pose_rope_shift[2]
        
        start_f = 1 if reference_latents is not None else 0
        f_pose = f - start_f
        
        shift_h = global_rope_H
        shift_w = global_rope_W
        
        freqs_pose = torch.cat([
            dit.freqs[0][start_f:f].view(f_pose, 1, 1, -1).expand(f_pose, h, w, -1),
            dit.freqs[1][shift_h:shift_h+h].view(1, h, 1, -1).expand(f_pose, h, w, -1),
            dit.freqs[2][shift_w:shift_w+w].view(1, 1, w, -1).expand(f_pose, h, w, -1)
        ], dim=-1)
        
        freqs_pose_real = torch.nn.functional.avg_pool2d(
            freqs_pose.real.permute(0, 3, 1, 2), kernel_size=2, stride=2
        ).permute(0, 2, 3, 1)
        
        freqs_pose_imag = torch.nn.functional.avg_pool2d(
            freqs_pose.imag.permute(0, 3, 1, 2), kernel_size=2, stride=2
        ).permute(0, 2, 3, 1)
        
        freqs_pose = torch.complex(freqs_pose_real, freqs_pose_imag)
        freqs_pose = freqs_pose.reshape(f_pose * (h//2) * (w//2), 1, -1)
        
        freqs = torch.cat([freqs, freqs_pose], dim=0)

    freqs = freqs.to(x.device)

    # VAP 
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # Reference frame detail transfer: adaptive strength based on timestep
    if enable_ref_detail_transfer:
        # Compute adaptive strength based on timestep
        # timestep is in [0, 1000]: high = noisy, low = clean
        t_val = timestep.item() / 1000.0  # normalize to [0, 1]
        # Only activate within the [ref_detail_transfer_end, ref_detail_transfer_start] range
        if t_val > ref_detail_transfer_start or t_val < ref_detail_transfer_end:
            enable_ref_detail_transfer = False
        else:
            # Linear interpolation: stronger when t is smaller (closer to clean image)
            progress = (ref_detail_transfer_start - t_val) / max(ref_detail_transfer_start - ref_detail_transfer_end, 1e-6)
            adaptive_scale = ref_detail_transfer_scale * progress
            # Number of spatial tokens per frame
            ref_hw = h * w
            # Default: apply to last 1/3 of layers (deeper layers carry more detail)
            if ref_detail_transfer_layers is None:
                # Priority 1: explicit cross_attn_guidance_layers (set by first_as_guidance_cross_attn).
                # This branch is required because first_as_guidance_cross_attn does NOT add
                # per-block attributes like `ref_detail_q`, so the hasattr-based fallback below
                # would wrongly return an empty list and silently disable the feature.
                if kwargs.get('first_as_guidance_cross_attn', False) and \
                        getattr(dit, 'cross_attn_guidance_layers', None) is not None:
                    ref_detail_transfer_layers = list(dit.cross_attn_guidance_layers)
                else:
                    ref_detail_transfer_layers = [
                        i for i, block in enumerate(dit.blocks)
                        if hasattr(block, 'ref_detail_q')
                    ]
                # print(f"Initialized ref_detail_transfer layers (auto) for blocks: {ref_detail_transfer_layers}");assert 0


    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]

    # Populate first-frame attention recorder geometry (f/h/w) so that
    # SelfAttention.forward can compute first-frame attention weights. f has
    # already been incremented by +1 for any prepended reference_latents; the
    # recorder will treat token index 0 as the "first frame" per RoPE ordering
    # (which may be the reference image if reference_latents is present).
    from ..models import wan_video_dit as _wvd_rec
    if _wvd_rec.FIRST_FRAME_ATTN_REC.get("enabled"):
        _wvd_rec.FIRST_FRAME_ATTN_REC["f"] = int(f)
        _wvd_rec.FIRST_FRAME_ATTN_REC["h"] = int(h)
        _wvd_rec.FIRST_FRAME_ATTN_REC["w"] = int(w)

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        # first_as_guidance_cross_attn: determine which blocks to apply concat self-attention
        enable_cross_attn_concat = kwargs.get('first_as_guidance_cross_attn', False) and enable_ref_detail_transfer
        cross_attn_num_ref_tokens = 0  # will be set on first concat

        for block_id, block in enumerate(dit.blocks):
            # first_as_guidance_cross_attn: prepend ref tokens before block forward
            if enable_cross_attn_concat and block_id in ref_detail_transfer_layers:
                x, freqs_for_block, cross_attn_num_ref_tokens = _prepare_cross_attn_concat(
                    x, ref_latent_tokens_for_detail, freqs, ref_hw)
            else:
                freqs_for_block = freqs

            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs_for_block, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs_for_block, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs_for_block, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs_for_block,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs_for_block,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs_for_block)

            # first_as_guidance_cross_attn: remove prepended ref tokens after block forward
            if enable_cross_attn_concat and block_id in ref_detail_transfer_layers:
                x = _remove_cross_attn_concat(x, cross_attn_num_ref_tokens)
            
            # print(x.shape);assert 0 # torch.Size([1, 32760, 5120])

            # Reference frame detail transfer: compute cross-attention from non-ref frames to ref frame
            # (not used when first_as_guidance_cross_attn is active, since concat self-attn replaces it)
            if enable_ref_detail_transfer and not enable_cross_attn_concat and block_id in ref_detail_transfer_layers:
                if kwargs.get('first_as_guidance_adaptive', False):
                    x = _apply_ref_detail_transfer_adaptive(x, block, f, h, w, ref_hw, adaptive_scale, ref_latent_tokens=ref_latent_tokens_for_detail)
                else:
                    x = _apply_ref_detail_transfer(x, block, f, h, w, ref_hw, adaptive_scale, ref_latent_tokens=ref_latent_tokens_for_detail)

            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    # print(x.shape);assert 0 # torch.Size([1, 32760, 64]) 21 52 30

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1

    if kwargs.get('temporal_concat'):
        x = x[:, :f*h*w]
        
    x = dit.unpatchify(x, (f, h, w))
    # print(x.shape);assert 0 # torch.Size([1, 16, 21, 104, 60]) 
    return x


def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
