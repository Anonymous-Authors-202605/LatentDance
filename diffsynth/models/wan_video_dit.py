import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
try:
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
    FLASH_ATTN_4_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_4_AVAILABLE = False

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
_flash_attention_logged = False


# ==============================================================================
# First-frame self-attention weight recorder
# ------------------------------------------------------------------------------
# Records the softmax(Q @ K^T / sqrt(d)) weights for query tokens in the FIRST
# frame attending to ALL key tokens. Used by the `first_frame_attn_vis` flag in
# the WanVideo pipeline to generate per-frame attention heatmaps that show how
# much each other frame influences the first frame's latent via self-attention.
#
# Protocol:
#   FIRST_FRAME_ATTN_REC["enabled"] = True/False  (master switch)
#   FIRST_FRAME_ATTN_REC["layers"]  = set/list of block indices to record
#                                     (None => all blocks)
#   FIRST_FRAME_ATTN_REC["f"], ["h"], ["w"] = patch grid shape (set by pipeline)
#   FIRST_FRAME_ATTN_REC["weights"] = { block_idx: [H_heads, f*h*w] averaged
#                                       attention weight from first-frame
#                                       queries to all key positions,
#                                       CPU tensor float32 }
#   FIRST_FRAME_ATTN_REC["per_block_layer_idx"] = currently-executing block idx
#     (set by DiTBlock before calling self_attn; SelfAttention.forward reads it)
# ==============================================================================
FIRST_FRAME_ATTN_REC = {
    "enabled": False,
    "layers": None,
    "f": None,
    "h": None,
    "w": None,
    "weights": {},
    "per_block_layer_idx": None,
}


def _record_first_frame_attn(q, k, num_heads, layer_idx):
    """Compute softmax attention weights from first-frame queries to all key
    positions and store a reduced [H_heads, f*h*w] tensor in the global
    recorder. Only runs when the global recorder is enabled and the layer_idx
    is selected.

    Args:
        q, k: tensors of shape [B, S, num_heads*head_dim] (after rope_apply).
              S = total_tokens = f*h*w (+ optional ref tokens at front).
        num_heads: number of attention heads.
        layer_idx: current block index.
    """
    rec = FIRST_FRAME_ATTN_REC
    if not rec.get("enabled"):
        return
    layers = rec.get("layers")
    if layers is not None and layer_idx not in layers:
        return
    f, h, w = rec.get("f"), rec.get("h"), rec.get("w")
    if f is None or h is None or w is None:
        return
    hw = h * w
    S = q.shape[1]
    # First-frame query tokens are at the BEGINNING of the sequence (frame 0
    # per RoPE ordering, possibly corresponding to the prepended reference
    # latent or the video's first frame depending on pipeline config).
    # We only consider the leading f*h*w tokens as the "video+ref" token grid;
    # trailing tokens (e.g. pose tokens appended by temporal_concat) are
    # included as keys but not as queries.
    expected = f * hw
    if S < expected:
        return
    offset = 0
    with torch.no_grad():
        # Reshape to per-head: [B, num_heads, S, head_dim]
        qh = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        kh = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        # First-frame query tokens: indices [offset, offset+hw)
        q_first = qh[:, :, offset:offset + hw, :].float()  # [B, n, hw, d]
        kh = kh.float()  # [B, n, S, d]
        head_dim = q_first.shape[-1]
        # Attention logits: [B, n, hw, S]
        # Chunk over query tokens to keep memory bounded
        B, n, _, _ = q_first.shape
        q_chunk = 256
        accum = torch.zeros(B, n, S, dtype=torch.float32, device=q.device)
        for i in range(0, hw, q_chunk):
            qc = q_first[:, :, i:i + q_chunk, :]  # [B, n, c, d]
            logits = torch.matmul(qc, kh.transpose(-1, -2)) / (head_dim ** 0.5)  # [B, n, c, S]
            weights = torch.softmax(logits, dim=-1)  # [B, n, c, S]
            # Average over query tokens in chunk
            accum = accum + weights.sum(dim=2)  # [B, n, S]
        accum = accum / float(hw)  # mean over all first-frame query tokens
        # Extract attention to the leading video-token positions only
        # [B, n, f*h*w]. Trailing tokens (pose/etc.) are ignored for the grid.
        weights_video = accum[:, :, offset:offset + expected]  # [B, n, f*h*w]
        # Reduce batch dim (usually 1)
        w_store = weights_video[0].detach().to(torch.float32).cpu()  # [n, f*h*w]
        rec["weights"][layer_idx] = w_store


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    global _flash_attention_logged
    if compatibility_mode:
        if not _flash_attention_logged:
            print("[flash_attention] Using: PyTorch scaled_dot_product_attention (compatibility_mode=True)")
            _flash_attention_logged = True
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_4_AVAILABLE:
        if not _flash_attention_logged:
            print("[flash_attention] Using: Flash Attention 4 (flash_attn.cute)")
            _flash_attention_logged = True
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_4_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        if not _flash_attention_logged:
            print("[flash_attention] Using: Flash Attention 3 (flash_attn_interface)")
            _flash_attention_logged = True
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        if not _flash_attention_logged:
            print("[flash_attention] Using: Flash Attention 2 (flash_attn)")
            _flash_attention_logged = True
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        if not _flash_attention_logged:
            print("[flash_attention] Using: SageAttention (sageattn)")
            _flash_attention_logged = True
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        if not _flash_attention_logged:
            print("[flash_attention] Using: PyTorch scaled_dot_product_attention (fallback)")
            _flash_attention_logged = True
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis





def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        # print(q.shape, k.shape, v.shape, freqs.shape);assert 0 
        # torch.Size([1, 56160, 5120]) torch.Size([1, 56160, 5120]) torch.Size([1, 56160, 5120]) torch.Size([28080, 1, 64])

        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        # First-frame attention weight recording (for first_frame_attn_vis).
        # Only runs when the global recorder is enabled; otherwise zero cost.
        if FIRST_FRAME_ATTN_REC.get("enabled"):
            layer_idx = FIRST_FRAME_ATTN_REC.get("per_block_layer_idx", None)
            if layer_idx is None:
                layer_idx = getattr(self, "_layer_idx", -1)
            _record_first_frame_attn(q, k, self.num_heads, layer_idx)

        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # Expose current layer index to the global first-frame attn recorder so
        # SelfAttention.forward can use it without extra argument wiring.
        if FIRST_FRAME_ATTN_REC.get("enabled"):
            FIRST_FRAME_ATTN_REC["per_block_layer_idx"] = getattr(self, "_layer_idx", -1)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        # Assign layer indices for the first-frame attention recorder so
        # DiTBlock / SelfAttention can tag recorded weights by block id.
        for _lid, _blk in enumerate(self.blocks):
            _blk._layer_idx = _lid
            _blk.self_attn._layer_idx = _lid
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        # depth_embedding is optionally initialized in train.py (like first_as_guidance_middle)
        # to avoid missing key errors when loading pretrained checkpoints
        self.depth_levels = 64
        self.depth_embedding = None

        # Keypoint index embeddings for fix_missing_warp_v2
        # Optionally initialized in train.py / test_batch.py when fix_missing_warp_v2 is enabled
        # kp_index_embedding_16ch: maps keypoint index -> 16ch embedding (replaces missing first-frame latent)
        # kp_index_embedding_4ch: maps keypoint index -> 4ch embedding (keypoint index spatial map)
        self.kp_index_embedding_16ch = None
        self.kp_index_embedding_4ch = None

        # TOKEN_REPLACE: optional trainable clone of `patch_embedding`, lazily initialized
        # in train.py / test.py when --token_replace is enabled. When present and when the
        # caller passes `token_replace=True` into `patchify`, the clone is used in place of
        # the frozen original `patch_embedding`. Kept as None to stay compatible with
        # pretrained checkpoints that do not carry this module.
        self.patch_embedding_token_replace = None

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, token_replace: bool = False):
        # TOKEN_REPLACE: when enabled, route patch embedding through a trainable clone
        # `patch_embedding_token_replace` instead of the frozen pretrained `patch_embedding`.
        # The clone is weight-initialized from `patch_embedding` in train.py / test.py so
        # that the first forward pass is numerically identical, and training updates only
        # the clone while the original projection and the rest of DiT stay frozen.
        if token_replace and getattr(self, "patch_embedding_token_replace", None) is not None:
            x = self.patch_embedding_token_replace(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def patchify_pose(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding_pose(x)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _build_freqs(self, f, h, w, device):
        """Build standard 3D RoPE frequencies (frame, height, width).
        
        Always uses the original 3D RoPE, preserving pretrained weight compatibility.
        Depth information is injected separately via learnable depth_embedding.
        
        Args:
            f, h, w: patch grid dimensions (after patchify)
            device: target device
        
        Returns:
            freqs: tensor [f*h*w, 1, D] complex RoPE frequencies
        """
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1)
        return freqs.to(device)

    def _apply_depth_embedding(self, x, f, h, w, depth_keypoints2):
        """Apply learnable depth embedding as additive bias on token embeddings.
        
        This preserves the original 3D RoPE (f, h, w) without any dimension reallocation,
        while allowing the model to learn depth-aware representations.
        
        Args:
            x: token embeddings [B, f*h*w, dim] (after patchify + rearrange)
            f, h, w: patch grid dimensions
            depth_keypoints2: int64 tensor [f, h, w] with values in [0, depth_levels-1],
                             pre-computed by LoadDepthKeypoints2 in the DataLoader.
        
        Returns:
            x: token embeddings with depth information added [B, f*h*w, dim]
        """
        if isinstance(depth_keypoints2, torch.Tensor):
            di = depth_keypoints2.long()
        else:
            di = torch.tensor(depth_keypoints2, dtype=torch.long, device=x.device)
        
        # Handle shape mismatch gracefully
        if di.shape != (f, h, w):
            di = torch.nn.functional.interpolate(
                di.float().unsqueeze(0).unsqueeze(0), size=(f, h, w), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        
        di = di.clamp(0, self.depth_levels - 1).to(x.device)
        
        # Flatten to [f*h*w] and lookup embedding
        depth_emb = self.depth_embedding(di.reshape(-1))  # [f*h*w, dim]
        
        # Add depth embedding to all batch elements
        x = x + depth_emb.unsqueeze(0).to(dtype=x.dtype)  # [B, f*h*w, dim]
        
        return x

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x, token_replace=bool(kwargs.get('token_replace', False)))
        
        # Always use standard 3D RoPE (preserves pretrained weight compatibility)
        freqs = self._build_freqs(f, h, w, x.device)
        
        # Inject depth information as additive embedding (does not modify RoPE)
        depth_keypoints2 = kwargs.get('depth_keypoints2', None)
        if depth_keypoints2 is not None and self.depth_embedding is not None:
            x = self._apply_depth_embedding(x, f, h, w, depth_keypoints2)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
