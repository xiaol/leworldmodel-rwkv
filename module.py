import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

try:
    from torch.utils.cpp_extension import CUDA_HOME, load as load_cpp_extension
except Exception:  # pragma: no cover - optional native extension support
    CUDA_HOME = None
    load_cpp_extension = None

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


def _ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise ValueError(f"Unsupported tensor shape for orthogonal init: {shape}")
        return x


def _pad_time(x, pad_len):
    if pad_len == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_len))


class _WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, chunk_len):
        B, T, H, C = w.shape
        if T % chunk_len != 0:
            raise ValueError(f"RWKV-7 CUDA sequence length {T} is not divisible by {chunk_len}")
        if not all(i.dtype == torch.float32 for i in [w, q, k, v, z, b]):
            raise TypeError("The upstream RWKV-7 wind_backstepping op currently expects fp32 inputs")

        w, q, k, v, z, b = [i.contiguous() for i in [w, q, k, v, z, b]]
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // chunk_len, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db, None


@lru_cache(maxsize=None)
def _load_wind_backstepping(head_size, chunk_len):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if hasattr(torch.ops.wind_backstepping, "forward"):
        return
    if load_cpp_extension is None:
        raise RuntimeError("torch.utils.cpp_extension.load is unavailable")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit was not found; install nvcc and set CUDA_HOME")

    source_dir = Path(os.environ.get("RWKV7_CUDA_SOURCE_DIR", Path(__file__).resolve().parent / "cuda"))
    sources = [source_dir / "wkv7_cuda_fp32.cu", source_dir / "wkv7_op_fp32.cpp"]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing RWKV-7 CUDA source file(s): {', '.join(missing)}")

    flags = [
        "-res-usage",
        f"-D_C_={head_size}",
        f"-D_CHUNK_LEN_={chunk_len}",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        "--extra-device-vectorization",
    ]
    load_cpp_extension(
        name=f"wind_backstepping_h{head_size}_c{chunk_len}",
        sources=[str(path) for path in sources],
        is_python_module=False,
        verbose=os.environ.get("RWKV7_CUDA_VERBOSE", "0") == "1",
        extra_cuda_cflags=flags,
    )
    if not hasattr(torch.ops.wind_backstepping, "forward"):
        raise RuntimeError("RWKV-7 CUDA extension loaded, but torch.ops.wind_backstepping.forward is unavailable")


def rwkv7_recurrence_torch(r, w, k, v, a, b, head_size):
    """RWKV-7 x070 time-mix recurrence.

    This is the upstream CUDA recurrence expressed in PyTorch. It is useful as a
    portable fallback, but it is not a fair speed benchmark for RWKV-7.
    """
    B, T, C = r.shape
    H = C // head_size
    dtype = r.dtype

    r = r.view(B, T, H, head_size)
    k = k.view(B, T, H, head_size)
    v = v.view(B, T, H, head_size)
    a = a.view(B, T, H, head_size)
    b = b.view(B, T, H, head_size)
    decay = torch.exp(-torch.exp(w.float()))
    decay = decay.view(B, T, H, head_size)

    state = torch.zeros(B, H, head_size, head_size, device=r.device, dtype=torch.float32)
    out = []
    for t in range(T):
        rt = r[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = a[:, t].float()
        bt = b[:, t].float()
        wt = decay[:, t]

        sa = (state * at.unsqueeze(-2)).sum(dim=-1)
        state = (
            state * wt.unsqueeze(-2)
            + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            + sa.unsqueeze(-1) * bt.unsqueeze(-2)
        )
        out.append((state * rt.unsqueeze(-2)).sum(dim=-1).to(dtype))

    return torch.stack(out, dim=1).reshape(B, T, C)


def rwkv7_recurrence_cuda(r, w, k, v, a, b, head_size, chunk_len):
    _load_wind_backstepping(head_size, chunk_len)

    B, T, C = r.shape
    H = C // head_size
    pad_len = (-T) % chunk_len
    T_padded = T + pad_len

    q, w, k, v, a, b = [
        _pad_time(t, pad_len).float().view(B, T_padded, H, head_size).contiguous()
        for t in [r, w, k, v, a, b]
    ]
    out = _WindBackstepping.apply(w, q, k, v, a, b, chunk_len)
    return out.reshape(B, T_padded, C)[:, :T].to(r.dtype)


def rwkv7_recurrence(r, w, k, v, a, b, head_size, backend="auto", chunk_len=16):
    if backend not in {"auto", "cuda", "torch"}:
        raise ValueError(f"Unknown RWKV-7 backend: {backend}")
    if backend == "torch" or not r.is_cuda:
        return rwkv7_recurrence_torch(r, w, k, v, a, b, head_size)

    try:
        return rwkv7_recurrence_cuda(r, w, k, v, a, b, head_size, chunk_len)
    except Exception as exc:
        if backend == "cuda":
            raise RuntimeError(
                "RWKV-7 CUDA backend is unavailable. Install the CUDA toolkit with "
                "nvcc, set CUDA_HOME if needed, and keep the upstream "
                "wind_backstepping sources under ./cuda or RWKV7_CUDA_SOURCE_DIR."
            ) from exc
        warnings.warn(
            f"Falling back to the slow PyTorch RWKV-7 recurrence because the native "
            f"wind_backstepping op is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return rwkv7_recurrence_torch(r, w, k, v, a, b, head_size)


class RWKV7TimeMix(nn.Module):
    """RWKV-7 x070 time-mix block adapted from blinkdl/rwkv-lm."""

    def __init__(self, dim, depth, layer_id, head_size=64, backend="auto", chunk_len=16):
        super().__init__()
        assert dim % head_size == 0, "hidden_dim must be divisible by RWKV head_size"
        if backend not in {"auto", "cuda", "torch"}:
            raise ValueError(f"Unknown RWKV-7 backend: {backend}")
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = dim // head_size
        self.backend = backend
        self.chunk_len = chunk_len

        ratio_0_to_1 = layer_id / max(depth - 1, 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / depth)
        ddd = torch.arange(dim, dtype=torch.float32).view(1, 1, dim) / dim

        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        linear = torch.arange(dim, dtype=torch.float32) / (dim - 1) - 0.5
        zigzag = (torch.arange(dim, dtype=torch.float32) % head_size)
        zigzag = (zigzag - ((head_size - 1) / 2)) / ((head_size - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        decay = -6 + 6 * (torch.arange(dim, dtype=torch.float32) / (dim - 1)) ** (
            1 + ratio_0_to_1**0.3
        )

        decay_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        aaa_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        gate_lora_dim = max(32, int(round((5 * (dim**0.5)) / 32) * 32))

        self.w1 = nn.Parameter(torch.zeros(dim, decay_lora_dim))
        self.w2 = nn.Parameter(_ortho_init(torch.zeros(decay_lora_dim, dim), 0.1))
        self.w0 = nn.Parameter((decay + 0.5 + zigzag * 2.5).view(1, 1, dim))

        self.a1 = nn.Parameter(torch.zeros(dim, aaa_lora_dim))
        self.a2 = nn.Parameter(_ortho_init(torch.zeros(aaa_lora_dim, dim), 0.1))
        self.a0 = nn.Parameter((torch.zeros(dim) - 0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, dim))

        if layer_id > 0:
            mv_lora_dim = max(32, int(round((1.7 * (dim**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(dim, mv_lora_dim))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(mv_lora_dim, dim), 0.1))
            self.v0 = nn.Parameter((torch.zeros(dim) + 0.73 - linear * 0.4).view(1, 1, dim))

        self.g1 = nn.Parameter(torch.zeros(dim, gate_lora_dim))
        self.g2 = nn.Parameter(_ortho_init(torch.zeros(gate_lora_dim, dim), 0.1))

        self.k_k = nn.Parameter((torch.zeros(dim) + 0.71 - linear * 0.1).view(1, 1, dim))
        self.k_a = nn.Parameter(torch.zeros(1, 1, dim) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(self.n_head, head_size) - 0.04)

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5 / (dim**0.5), 0.5 / (dim**0.5))
        self.key.weight.data.uniform_(-0.05 / (dim**0.5), 0.05 / (dim**0.5))
        self.value.weight.data.uniform_(-0.5 / (dim**0.5), 0.5 / (dim**0.5))
        self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        xx = F.pad(x, (0, 0, 1, -1)) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        w = -F.softplus(-w) - 0.5
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, self.n_head, self.head_size), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = rwkv7_recurrence(
            r,
            w,
            k,
            v,
            -kk,
            kk * a,
            self.head_size,
            backend=self.backend,
            chunk_len=self.chunk_len,
        )
        x = self.ln_x(x.reshape(B * T, C)).view(B, T, C)
        x = x + (
            (r.view(B, T, self.n_head, self.head_size) * k.view(B, T, self.n_head, self.head_size) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, self.n_head, self.head_size)
        ).view(B, T, C)
        return self.output(x * g), v_first


class RWKV7ChannelMix(nn.Module):
    def __init__(self, dim, depth, layer_id, ffn_dim=None):
        super().__init__()
        ratio_1_to_almost0 = 1.0 - (layer_id / depth)
        ddd = torch.arange(dim, dtype=torch.float32).view(1, 1, dim) / dim
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        ffn_dim = ffn_dim or int((dim * 3.5) // 32 * 32)
        self.key = nn.Linear(dim, ffn_dim, bias=False)
        self.value = nn.Linear(ffn_dim, dim, bias=False)

        self.key.weight.data.uniform_(-0.5 / (dim**0.5), 0.5 / (dim**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = F.pad(x, (0, 0, 1, -1)) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class RWKV7Block(nn.Module):
    def __init__(self, dim, depth, layer_id, head_size=64, ffn_dim=None, backend="auto", chunk_len=16):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = RWKV7TimeMix(
            dim,
            depth,
            layer_id,
            head_size=head_size,
            backend=backend,
            chunk_len=chunk_len,
        )
        self.ffn = RWKV7ChannelMix(dim, depth, layer_id, ffn_dim=ffn_dim)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7(nn.Module):
    """RWKV-7 x070 sequence model for continuous LeWM embeddings."""

    def __init__(self, dim, depth, head_size=64, ffn_dim=None, backend="auto", chunk_len=16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RWKV7Block(
                    dim,
                    depth,
                    i,
                    head_size=head_size,
                    ffn_dim=ffn_dim,
                    backend=backend,
                    chunk_len=chunk_len,
                )
                for i in range(depth)
            ]
        )
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x):
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.ln_out(x)


class RWKV7Predictor(nn.Module):
    """Autoregressive LeWM predictor using RWKV-7 x070 instead of attention."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads=None,
        mlp_dim=None,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        backend="auto",
        chunk_len=16,
    ):
        super().__init__()
        del heads, dropout
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.rwkv = RWKV7(
            hidden_dim,
            depth,
            head_size=dim_head,
            ffn_dim=mlp_dim,
            backend=backend,
            chunk_len=chunk_len,
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != (output_dim or input_dim)
            else nn.Identity()
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = self.input_proj(x) + self.cond_proj(c)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.output_proj(self.rwkv(x))
