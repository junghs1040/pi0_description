# π₀ 모델 데이터 흐름 Step-by-Step 완전 가이드

이 문서는 π₀ 모델에서 **입력 데이터가 어떻게 처리되어 최종 출력이 되는지**를 한 단계씩 추적합니다.

> **📌 학습 vs 추론 구분**
> 이 문서는 **학습(Training)** 과정을 기본으로 설명하며, 각 Step에서 추론(Inference)과 차이가 있는 경우 `🔄 학습 vs 추론` 박스로 표시합니다.
> - 🏋️ **학습**: Ground truth actions + noise → Flow Matching loss 계산
> - 🎯 **추론**: Pure noise에서 시작 → 10회 Euler integration으로 action 생성

Step 0: 원본 입력 데이터 (Images, State, Text, Actions)
Step 1: Observation 객체 생성 (uint8 → float32 정규화)
Step 2: Image Embedding (SigLIP) - 3×256 = 768 tokens
Step 3: Text Embedding (Gemma Embedder) - 16 tokens
Step 4: Prefix Concatenation - 784 tokens (Image + Text)
Step 5: Action Embedding (Suffix) - 32 tokens + Flow Matching
Step 6: Attention Mask 생성 - [4, 816, 816]
Step 7: Transformer Layer 0 상세 분석
7-1: Pre-Attention RMSNorm (AdaRMS)
7-2: QKV Projection (Multi-Expert)
7-3: RoPE (Rotary Position Embedding)
7-4: Grouped Query Attention
7-5: Output Projection (Expert별)
7-6: Gated Residual
7-7: FeedForward Network
Step 8: Transformer Layers 1-17 (18 layers total)
Step 9: Final Layer Normalization
Step 10-11: Velocity Prediction + Flow Matching Loss


**예시 데이터**:
- Batch Size: B = 4
- Images: 3개 (base_0, left_wrist_0, right_wrist_0)
- Text: 16 tokens
- Actions: 32 timesteps, 7 DoF
- Model: π₀.₅ (with AdaRMS)

---

## 📍 Step 0: 원본 입력 데이터

```python
# ═══════════════════════════════════════════════════════════════
# Step 0: Raw Input (Python Dictionary)
# ═══════════════════════════════════════════════════════════════

raw_input = {
    # ─── Images ───
    "image": {
        "base_0_rgb": np.array([4, 224, 224, 3], dtype=uint8),        # [0, 255]
        "left_wrist_0_rgb": np.array([4, 224, 224, 3], dtype=uint8),
        "right_wrist_0_rgb": np.array([4, 224, 224, 3], dtype=uint8),
    },
    "image_mask": {
        "base_0_rgb": np.array([True, True, True, True]),
        "left_wrist_0_rgb": np.array([True, True, True, True]),
        "right_wrist_0_rgb": np.array([True, True, True, True]),
    },

    # ─── Robot State ───
    "state": np.array([4, 7], dtype=float32),  # [x, y, z, qx, qy, qz, gripper]

    # ─── Language Command ───
    "tokenized_prompt": np.array([
        [15234, 67, 123, 9876, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "pick up fork" + padding
        [8921, 456, 789, 234, 567, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "grasp red cup" + padding
        [...],
        [...],
    ], dtype=int32),  # [4, 16]
    "tokenized_prompt_mask": np.array([
        [True, True, True, True, False, False, ...],  # 첫 4개만 valid
        [True, True, True, True, True, False, ...],   # 첫 5개만 valid
        [...],
        [...],
    ], dtype=bool),  # [4, 16]

    # ─── Actions (Training only) ───
    "actions": np.array([4, 32, 7], dtype=float32),  # Ground truth actions
}
```

> **🔄 학습 vs 추론**
> | | 학습 (Training) | 추론 (Inference) |
> |---|---|---|
> | **Images** | 동일 | 동일 |
> | **State** | 동일 | 동일 |
> | **Text** | 동일 | 동일 |
> | **Actions** | ✅ Ground truth 필요 | ❌ 없음 (noise에서 생성) |

---

## 📍 Step 1: Observation 객체 생성

**코드 위치**: `src/openpi/models/model.py:110-125`

```python
# ═══════════════════════════════════════════════════════════════
# Step 1: Dictionary → Observation Object
# ═══════════════════════════════════════════════════════════════

observation = Observation.from_dict(raw_input)

# ─── 내부 처리 ───
# 1. uint8 이미지를 float32 [-1, 1]로 정규화
for key in raw_input["image"]:
    image = raw_input["image"][key]  # [4, 224, 224, 3] uint8 [0, 255]
    image = image.astype(float32) / 255.0 * 2.0 - 1.0  # float32 [-1, 1]

# 2. 구조화된 Observation 객체 생성
observation = Observation(
    images={
        "base_0_rgb": [4, 224, 224, 3],        # float32, [-1, 1]
        "left_wrist_0_rgb": [4, 224, 224, 3],
        "right_wrist_0_rgb": [4, 224, 224, 3],
    },
    image_masks={
        "base_0_rgb": [4],        # bool
        "left_wrist_0_rgb": [4],
        "right_wrist_0_rgb": [4],
    },
    state=[4, 7],                              # float32
    tokenized_prompt=[4, 16],                  # int32
    tokenized_prompt_mask=[4, 16],             # bool
)

# ✅ Output Shape:
# - Images: 3개 × [4, 224, 224, 3] float32 [-1, 1]
# - State: [4, 7] float32
# - Text: [4, 16] int32
```

---

## 📍 Step 2: Image Embedding (SigLIP)

**코드 위치**: `src/openpi/models/pi0.py:113-125` + `src/openpi/models/siglip.py`

```python
# ═══════════════════════════════════════════════════════════════
# Step 2: Images → Image Tokens (SigLIP Vision Encoder)
# ═══════════════════════════════════════════════════════════════

# 각 이미지마다 독립적으로 처리
image_tokens_list = []
for image_name in observation.images:
    image = observation.images[image_name]  # [4, 224, 224, 3]

    # ─── SigLIP Forward Pass ───
    image_tokens, _ = self.PaliGemma.img(image, train=False)

    # ─── SigLIP 내부 처리 ───
    # 2-1. Patch Embedding
    # Image [4, 224, 224, 3] → 14×14 patches
    patches = einops.rearrange(
        image,
        'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
        p1=14, p2=14
    )
    # patches: [4, 256, 588]  (256 = 16×16, 588 = 14×14×3)

    patch_emb = nn.Dense(1152)(patches)  # [4, 256, 1152]

    # 2-2. Positional Embedding (Sinusoidal 2D)
    h, w = 16, 16  # 224/14 = 16
    y, x = jnp.mgrid[:h, :w]  # [16, 16]
    omega = jnp.arange(1152 // 4) / (1152 // 4 - 1)
    omega = 1.0 / (10000 ** omega)

    y_emb = jnp.einsum("m,d->md", y.flatten(), omega)  # [256, 288]
    x_emb = jnp.einsum("m,d->md", x.flatten(), omega)  # [256, 288]
    pos_emb = jnp.concatenate([
        jnp.sin(x_emb), jnp.cos(x_emb),
        jnp.sin(y_emb), jnp.cos(y_emb)
    ], axis=1)  # [256, 1152]

    x = patch_emb + pos_emb[None, :, :]  # [4, 256, 1152]

    # 2-3. Transformer Encoder (12 layers)
    for layer in range(12):
        # Pre-Norm
        x_norm = nn.LayerNorm()(x)

        # Multi-Head Self-Attention
        q = k = v = x_norm  # [4, 256, 1152]
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=16,  # 1152 / 16 = 72 per head
        )(q, k)  # [4, 256, 1152]

        x = x + attn_out  # Residual

        # Pre-Norm
        x_norm = nn.LayerNorm()(x)

        # MLP
        mlp_out = nn.Dense(4608)(x_norm)  # [4, 256, 4608]
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(1152)(mlp_out)  # [4, 256, 1152]

        x = x + mlp_out  # Residual

    # 2-4. Final Projection to Gemma dimension
    image_tokens = nn.Dense(2048)(x)  # [4, 256, 2048]

    image_tokens_list.append(image_tokens)

# ✅ Output:
# image_tokens_list = [
#     [4, 256, 2048],  # base_0_rgb
#     [4, 256, 2048],  # left_wrist_0_rgb
#     [4, 256, 2048],  # right_wrist_0_rgb
# ]
# Total: 3 × 256 = 768 image tokens
```

---

## 📍 Step 3: Text Embedding (Gemma Embedder)

**코드 위치**: `src/openpi/models/gemma.py:148-154` + `pi0.py:128-133`

```python
# ═══════════════════════════════════════════════════════════════
# Step 3: Token IDs → Text Embeddings
# ═══════════════════════════════════════════════════════════════

# 입력: observation.tokenized_prompt
# [4, 16] int32
tokenized_prompt = observation.tokenized_prompt

# ─── Embedder.encode() ───
tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")

# ─── Embedder 내부 ───
class Embedder:
    def encode(self, x):
        # 1. Embedding table lookup
        # input_embedding_table: [257152, 2048]
        x = self.input_embedding_table[(x,)]  # [4, 16, 2048]

        # 2. Scale by √embed_dim (Attention Is All You Need 논문)
        x *= jnp.sqrt(2048)  # ≈ 45.25
        # 이유: Embedding 값의 scale을 조정하여 position encoding과 균형

        return x  # [4, 16, 2048]

# ✅ Output:
# text_tokens: [4, 16, 2048]
#
# 예시 변환:
# Token ID 15234 → embedding_table[15234] → [2048 dims] × 45.25
# Token ID 67    → embedding_table[67]    → [2048 dims] × 45.25
# Token ID 0     → embedding_table[0]     → [2048 dims] × 45.25 (padding)
```

---

## 📍 Step 4: Prefix Token Concatenation

**코드 위치**: `src/openpi/models/pi0.py:106-137`

```python
# ═══════════════════════════════════════════════════════════════
# Step 4: Image + Text → Prefix Sequence
# ═══════════════════════════════════════════════════════════════

def embed_prefix(self, obs):
    tokens = []
    input_mask = []
    ar_mask = []

    # ─── 4-1: Image tokens 추가 ───
    for name in obs.images:
        image_tokens = image_tokens_list.pop(0)  # [4, 256, 2048]
        tokens.append(image_tokens)

        # Mask: 모든 image token은 valid
        mask = einops.repeat(
            obs.image_masks[name],  # [4]
            "b -> b s",
            s=256
        )  # [4, 256]
        input_mask.append(mask)

        # AR Mask: image는 bidirectional attention
        ar_mask += [False] * 256

    # ─── 4-2: Text tokens 추가 ───
    if obs.tokenized_prompt is not None:
        text_tokens = tokenized_inputs  # [4, 16, 2048]
        tokens.append(text_tokens)
        input_mask.append(obs.tokenized_prompt_mask)  # [4, 16]

        # AR Mask: text도 bidirectional attention
        ar_mask += [False] * 16

    # ─── 4-3: Concatenation ───
    prefix_tokens = jnp.concatenate(tokens, axis=1)
    # [4, 768, 2048] + [4, 16, 2048] = [4, 784, 2048]

    prefix_mask = jnp.concatenate(input_mask, axis=1)  # [4, 784]
    prefix_ar_mask = jnp.array(ar_mask)  # [784]

    return prefix_tokens, prefix_mask, prefix_ar_mask

# ✅ Output:
# - prefix_tokens: [4, 784, 2048]
#   ├─ Image 0: tokens[0:256]
#   ├─ Image 1: tokens[256:512]
#   ├─ Image 2: tokens[512:768]
#   └─ Text:    tokens[768:784]
# - prefix_mask: [4, 784] (all True)
# - prefix_ar_mask: [784] (all False = bidirectional)
```

---

## 📍 Step 5: Action Embedding (Suffix)

**코드 위치**: `src/openpi/models/pi0.py:139-186`

```python
# ═══════════════════════════════════════════════════════════════
# Step 5: Actions → Action Tokens (Suffix)
# ═══════════════════════════════════════════════════════════════

# Training input: ground truth actions
actions = raw_input["actions"]  # [4, 32, 7]

# ─── 5-1: Flow Matching Preparation ───
# Noise 생성
rng, noise_rng, time_rng = jax.random.split(rng, 3)
noise = jax.random.normal(noise_rng, actions.shape)  # [4, 32, 7]

# Timestep 샘플링 (Beta distribution)
time = jax.random.beta(time_rng, 1.5, 1.0, batch_shape=[4])
# time: [4]  예: [0.234, 0.891, 0.456, 0.123]

# Flow interpolation: x_t = t·noise + (1-t)·actions
# - t=1: pure noise
# - t=0: real actions
# - 0<t<1: interpolated
time_expanded = time[:, None, None]  # [4, 1, 1]
x_t = time_expanded * noise + (1 - time_expanded) * actions
# x_t: [4, 32, 7]

# ─── 5-2: Action Token Projection ───
action_tokens = self.action_in_proj(x_t)
# Linear(in=7, out=2048)
# action_tokens: [4, 32, 2048]

# ─── 5-3: Timestep Embedding (Sinusoidal) ───
def posemb_sincos(pos, embedding_dim, min_period, max_period):
    # pos: [4]
    # embedding_dim: 2048

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # [1024]
    period = min_period * (max_period / min_period) ** fraction
    # period: [0.004, ..., 4.0]  (1024 values)

    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,  # [4]
        1.0 / period * 2 * jnp.pi  # [1024]
    )  # [4, 1024]

    emb = jnp.concatenate([
        jnp.sin(sinusoid_input),  # [4, 1024]
        jnp.cos(sinusoid_input),  # [4, 1024]
    ], axis=-1)  # [4, 2048]

    return emb

time_emb = posemb_sincos(time, 2048, min_period=4e-3, max_period=4.0)
# time_emb: [4, 2048]

# ─── 5-4: π₀.₅ AdaRMS Conditioning ───
# Time MLP for AdaRMS
time_emb = self.time_mlp_in(time_emb)   # Linear(2048 → 2048)
time_emb = nnx.swish(time_emb)          # Swish activation
time_emb = self.time_mlp_out(time_emb)  # Linear(2048 → 2048)
time_emb = nnx.swish(time_emb)
# time_emb: [4, 2048]

action_expert_tokens = action_tokens  # [4, 32, 2048]
adarms_cond = time_emb  # [4, 2048]  ← AdaRMS에서 사용

# ─── 5-5: Suffix 구성 ───
suffix_tokens = action_expert_tokens  # [4, 32, 2048]
suffix_mask = jnp.ones([4, 32], dtype=bool)  # 모두 valid

# AR Mask: First token is causal boundary, rest can attend to each other
suffix_ar_mask = jnp.array([True] + [False] * 31)
# [True, False, False, ..., False]
#  ^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
# Causal  Bidirectional within action block

# ✅ Output:
# - suffix_tokens: [4, 32, 2048]
# - suffix_mask: [4, 32] (all True)
# - suffix_ar_mask: [32] ([True, False, False, ...])
# - adarms_cond: [4, 2048] (for AdaRMS conditioning)
```

> **🔄 학습 vs 추론**
> | | 학습 (Training) | 추론 (Inference) |
> |---|---|---|
> | **입력 actions** | Ground truth actions | 없음 (noise에서 시작) |
> | **Noise** | `noise ~ N(0, I)` | `noise ~ N(0, I)` (= 초기 x_t) |
> | **Timestep t** | `t ~ Beta(1.5, 1.0)` 랜덤 샘플링 | `t = 1.0, 0.9, ..., 0.1` 고정 스케줄 |
> | **x_t 계산** | `x_t = t·noise + (1-t)·actions` (interpolation) | 반복마다 Euler step으로 업데이트 |
> | **횟수** | **1회** (한 번의 forward pass) | **10회** 반복 (매번 suffix 재생성) |
>
> 학습에서는 랜덤 t로 interpolated sample을 만들지만, 추론에서는 t=1.0(pure noise)에서 시작하여 매 step마다 velocity를 예측하고 `x_{t+dt} = x_t + dt·v_t`로 업데이트합니다.

---

## 📍 Step 6: Attention Mask 생성

**코드 위치**: `src/openpi/models/pi0.py:19-44` + `202-208`

```python
# ═══════════════════════════════════════════════════════════════
# Step 6: Create Attention Mask
# ═══════════════════════════════════════════════════════════════

# ─── 6-1: Concatenate masks ───
input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
# [4, 784] + [4, 32] = [4, 816]

ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
# [784] + [32] = [816]
# ar_mask = [False, False, ..., False, True, False, False, ..., False]
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
#            Prefix (bidirectional)        |     Suffix (bidirectional)
#                                      Causal boundary

# ─── 6-2: Generate Attention Mask ───
def make_attn_mask(input_mask, mask_ar):
    # input_mask: [4, 816]
    # mask_ar: [816]

    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # [4, 816]

    # Cumulative sum: marks causal boundaries
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # Example for one sample:
    # cumsum = [0, 0, ..., 0, 1, 1, 1, ..., 1]
    #          ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
    #          Prefix (0)    Suffix (1)

    # Create causal mask
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # [4, 1, 816] <= [4, 816, 1] → [4, 816, 816]

    # Apply padding mask
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    # [4, 816, 816]

    return jnp.logical_and(attn_mask, valid_mask)

attn_mask = make_attn_mask(input_mask, ar_mask)
# attn_mask: [4, 816, 816]

# ─── 6-3: Attention Mask 시각화 ───
"""
Attention pattern for one sample [816, 816]:

              Prefix(784)              Suffix(32)
         ┌────────────────────────┬──────────────┐
         │ Img0  Img1  Img2  Text │ Act1 ... Act32│
         │ 0-255 256-  512-  768- │ 784  ... 815  │
         │       511   767   783  │               │
    ─────┼────────────────────────┼──────────────┤
Prefix   │  ✓     ✓     ✓     ✓  │  ✓  ...  ✓   │ ← Prefix는 모든 것을
0-783    │  ✓     ✓     ✓     ✓  │  ✓  ...  ✓   │   볼 수 있음
         │  ...                   │  ...          │   (Bidirectional)
    ─────┼────────────────────────┼──────────────┤
Suffix   │  ✓     ✓     ✓     ✓  │  ✓   ✗   ✗   │ ← Suffix는 prefix는
784      │  ✓     ✓     ✓     ✓  │  ✓   ✓   ✗   │   볼 수 있지만,
785      │  ✓     ✓     ✓     ✓  │  ✓   ✓   ✓   │   suffix 내에서는
...      │  ...                   │  ...          │   causal
815      │  ✓     ✓     ✓     ✓  │  ✓   ✓   ✓   │
         └────────────────────────┴──────────────┘
                                     ^^^^^^^^^^
                                     Causal mask
"""

# ─── 6-4: Position Encoding ───
positions = jnp.cumsum(input_mask, axis=1) - 1
# [4, 816]
# Example: [[0, 1, 2, ..., 783, 784, 785, ..., 815], ...]

# ✅ Output:
# - attn_mask: [4, 816, 816] bool
# - positions: [4, 816] int32
```

---

## 📍 Step 7: Multi-Expert Transformer Layer 0

이제 18개의 Transformer layer 중 **첫 번째 layer**를 자세히 봅니다.

**코드 위치**: `src/openpi/models/gemma.py:284-333`

### Step 7-1: Pre-Attention RMSNorm

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-1: Pre-Attention RMSNorm (with AdaRMS)
# ═══════════════════════════════════════════════════════════════

# 입력:
xs = [prefix_tokens, suffix_tokens]
# xs[0]: [4, 784, 2048]  (Expert 0 - PaliGemma)
# xs[1]: [4, 32, 2048]   (Expert 1 - Action)

adarms_cond = [None, adarms_cond]
# adarms_cond[0]: None (Expert 0는 conditioning 안 함)
# adarms_cond[1]: [4, 2048] (Expert 1은 timestep conditioning)

# ─── RMSNorm 적용 ───
pre_attn = []
gates = []

for i, x in enumerate(xs):
    if x is not None:
        # 1. Root Mean Square 계산
        var = jnp.mean(jnp.square(x.astype(float32)), axis=-1, keepdims=True)
        # xs[0]: [4, 784, 2048] → var: [4, 784, 1]
        # xs[1]: [4, 32, 2048]  → var: [4, 32, 1]

        # 2. Normalization
        normed_inputs = x * jnp.reciprocal(jnp.sqrt(var + 1e-6))
        # xs[0]: [4, 784, 2048]
        # xs[1]: [4, 32, 2048]

        # 3. Expert별 처리
        if adarms_cond[i] is None:  # Expert 0 (Prefix)
            # ─── Regular RMSNorm ───
            scale = self.param("scale", zeros_init(), (2048,))  # [2048]
            x_norm = normed_inputs * (1 + scale)
            gate = None

        else:  # Expert 1 (Suffix) - π₀.₅
            # ─── Adaptive RMSNorm (AdaRMS) ───
            # Modulation network
            modulation = nn.Dense(2048 * 3)(adarms_cond[i])
            # Input: [4, 2048] → Output: [4, 6144]

            # Split into scale, shift, gate
            scale, shift, gate = jnp.split(modulation, 3, axis=-1)
            # scale: [4, 2048]
            # shift: [4, 2048]
            # gate:  [4, 2048]

            # AdaIN (Adaptive Instance Normalization) style
            scale_expanded = scale[:, None, :]  # [4, 1, 2048]
            shift_expanded = shift[:, None, :]  # [4, 1, 2048]

            x_norm = normed_inputs * (1 + scale_expanded) + shift_expanded
            # [4, 32, 2048] * [4, 1, 2048] + [4, 1, 2048]
            # → [4, 32, 2048]

        pre_attn.append(x_norm)
        gates.append(gate)

# ✅ Output:
# pre_attn[0]: [4, 784, 2048]  (Normalized prefix, no conditioning)
# pre_attn[1]: [4, 32, 2048]   (Normalized suffix, with timestep conditioning)
# gates[0]: None
# gates[1]: [4, 2048] (for gated residual later)
```

### Step 7-2: QKV Projection (Multi-Expert)

**코드 위치**: `src/openpi/models/gemma.py:158-199`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-2: QKV Projection (Expert-specific Weights)
# ═══════════════════════════════════════════════════════════════

qkvs = []

for i, (x, config) in enumerate(zip(pre_attn, configs)):
    if x is None:
        continue

    # ─── Grouped Query Attention (GQA) ───
    # num_heads = 8, num_kv_heads = 1, head_dim = 256

    # Query Projection
    q_einsum = lora.Einsum(
        shape=(8, 2048, 256),  # (num_heads, width, head_dim)
        name=_name("q_einsum", i),  # "q_einsum" or "q_einsum_1"
        ...
    )
    q = q_einsum("BTD,NDH->BTNH", x)
    # x[0]: [4, 784, 2048] → q[0]: [4, 784, 8, 256]
    # x[1]: [4, 32, 2048]  → q[1]: [4, 32, 8, 256]

    # Key/Value Projection (shared, only 1 head for GQA)
    kv_einsum = lora.Einsum(
        shape=(2, 1, 2048, 256),  # (2, num_kv_heads, width, head_dim)
        name=_name("kv_einsum", i),  # "kv_einsum" or "kv_einsum_1"
        ...
    )
    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
    # x[0]: [4, 784, 2048] → k[0], v[0]: [4, 784, 1, 256]
    # x[1]: [4, 32, 2048]  → k[1], v[1]: [4, 32, 1, 256]

    qkvs.append((q, k, v))

# ✅ Output:
# qkvs[0]: (q[4,784,8,256], k[4,784,1,256], v[4,784,1,256]) ← Expert 0 weight
# qkvs[1]: (q[4,32,8,256],  k[4,32,1,256],  v[4,32,1,256])  ← Expert 1 weight
#          ^^^ 완전히 독립적인 weight 사용!
```

### Step 7-3: RoPE (Rotary Position Embedding)

**코드 위치**: `src/openpi/models/gemma.py:424-440`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-3: Apply RoPE to Q and K
# ═══════════════════════════════════════════════════════════════

# Concatenate QKV from all experts
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs))
# q: [4, 816, 8, 256]  (784 + 32)
# k: [4, 816, 1, 256]
# v: [4, 816, 1, 256]

def _apply_rope(x, positions, max_wavelength=10_000):
    # x: [4, 816, H, 256]
    # positions: [4, 816]

    # Frequency 계산
    freq_exponents = (2.0 / 256) * jnp.arange(256 // 2)  # [128]
    timescale = max_wavelength ** freq_exponents
    # timescale: [10000^0, ..., 10000^(254/256)]

    # Position에 따른 radians
    radians = positions[..., None] / timescale[None, None, :]
    # [4, 816, 1] / [1, 1, 128] = [4, 816, 128]
    radians = radians[..., None, :]  # [4, 816, 1, 128]

    # Sin/Cos 계산
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    # sin, cos: [4, 816, 1, 128]

    # Split features into two halves
    x1, x2 = jnp.split(x, 2, axis=-1)
    # x1, x2: [4, 816, H, 128]

    # Rotation
    res = jnp.concatenate([
        x1 * cos - x2 * sin,  # [4, 816, H, 128]
        x2 * cos + x1 * sin,  # [4, 816, H, 128]
    ], axis=-1)  # [4, 816, H, 256]

    return res

# Apply RoPE
q = _apply_rope(q, positions=positions)  # [4, 816, 8, 256]
k = _apply_rope(k, positions=positions)  # [4, 816, 1, 256]

# Scale Q by 1/√head_dim
q *= 256 ** -0.5  # ≈ 0.0625

# ✅ Output:
# q: [4, 816, 8, 256] (with positional info)
# k: [4, 816, 1, 256] (with positional info)
# v: [4, 816, 1, 256] (no change)
```

### Step 7-4: Grouped Query Attention

**코드 위치**: `src/openpi/models/gemma.py:216-231`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-4: Grouped Query Attention (GQA)
# ═══════════════════════════════════════════════════════════════

# Reshape Q for Grouped Query Attention
q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=1)
# [4, 816, 8, 256] → [4, 816, 1, 8, 256]
#  B   T   N   H       B   T   K  G  H
# N = K × G (num_heads = num_kv_heads × group_size)
# 8 = 1 × 8

# Attention scores
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
# q: [4, 816, 1, 8, 256]
# k: [4, 816, 1, 256]
# logits: [4, 1, 8, 816, 816]
#         B  K  G   T    S

# Apply attention mask
big_neg = -2.3819763e38  # Large negative value
attn_mask_expanded = attn_mask[:, None, None, :, :]
# [4, 816, 816] → [4, 1, 1, 816, 816]

masked_logits = jnp.where(attn_mask_expanded, logits, big_neg)
# masked_logits: [4, 1, 8, 816, 816]

# Softmax (in float32 for stability)
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
# probs: [4, 1, 8, 816, 816]

# Apply to values
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
# probs: [4, 1, 8, 816, 816]
# v: [4, 816, 1, 256]
# encoded: [4, 816, 1, 8, 256]

# Reshape back
encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
# [4, 816, 1, 8, 256] → [4, 816, 8, 256]

# ✅ Output:
# encoded: [4, 816, 8, 256]  (attention-weighted values)
```

### Step 7-5: Output Projection (Multi-Expert)

**코드 위치**: `src/openpi/models/gemma.py:233-249`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-5: Output Projection (Expert별 독립)
# ═══════════════════════════════════════════════════════════════

# encoded: [4, 816, 8, 256] (모든 토큰의 attention output)

out = []
start = 0

for i, (x, config) in enumerate(zip(xs, configs)):
    if x is not None:
        end = start + x.shape[1]
        # Expert 0: start=0, end=784
        # Expert 1: start=784, end=816

        # Expert별 독립적인 output projection
        out_einsum = lora.Einsum(
            shape=(8, 256, 2048),  # (num_heads, head_dim, width)
            name=_name("attn_vec_einsum", i),  # "attn_vec_einsum" or "_1"
            ...
        )

        # Slice and project
        expert_encoded = encoded[:, start:end]
        # Expert 0: [4, 784, 8, 256]
        # Expert 1: [4, 32, 8, 256]

        expert_out = out_einsum("BTNH,NHD->BTD", expert_encoded)
        # Expert 0: [4, 784, 2048]
        # Expert 1: [4, 32, 2048]

        out.append(expert_out)
        start = end
    else:
        out.append(None)

# ✅ Output:
# out[0]: [4, 784, 2048]  (Prefix, Expert 0 weight 사용)
# out[1]: [4, 32, 2048]   (Suffix, Expert 1 weight 사용)
```

### Step 7-6: Gated Residual Connection

**코드 위치**: `src/openpi/models/gemma.py:309-312` + `453-459`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-6: Gated Residual (AdaRMS gate)
# ═══════════════════════════════════════════════════════════════

def _gated_residual(x, y, gate):
    if x is None or y is None:
        return None
    if gate is None:
        return x + y  # Regular residual
    return x + y * gate  # Gated residual

xs = [
    _gated_residual(xs[0], out[0], gates[0]),
    _gated_residual(xs[1], out[1], gates[1]),
]

# Expert 0 (Prefix):
# xs[0] = prefix_tokens + out[0]
# [4, 784, 2048] + [4, 784, 2048] = [4, 784, 2048]

# Expert 1 (Suffix) - with AdaRMS gate:
# xs[1] = suffix_tokens + out[1] * gates[1]
# [4, 32, 2048] + [4, 32, 2048] * [4, 1, 2048] = [4, 32, 2048]
#                                   ^^^^^^^
#                                   Gate controls residual strength

# ✅ Output:
# xs[0]: [4, 784, 2048]  (after first residual)
# xs[1]: [4, 32, 2048]   (after gated residual)
```

### Step 7-7: FeedForward Network

**코드 위치**: `src/openpi/models/gemma.py:314-330`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-7: FeedForward Network (Expert별 독립)
# ═══════════════════════════════════════════════════════════════

out = []
gates = []

for i, (x, config) in enumerate(zip(xs, configs)):
    if x is not None:
        # ─── Pre-FFN RMSNorm ───
        x_norm, gate = RMSNorm(name=_name("pre_ffw_norm", i))(
            x, adarms_cond[i]
        )
        # Same AdaRMS logic as before
        # x_norm: [4, 784, 2048] or [4, 32, 2048]
        # gate: None or [4, 2048]

        # ─── FeedForward ───
        # Gated FFN (SwiGLU variant)
        w_gating = self.param(
            _name("gating_einsum", i),
            ...,
            (2, 2048, 16384)  # (2, features, mlp_dim)
        )

        # Two projections
        ff_gate = jnp.dot(x_norm, w_gating[0])  # [B, T, 16384]
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x_norm, w_gating[1])  # [B, T, 16384]
        activations = gate_value * ff1  # Element-wise multiply

        # Output projection
        w_linear = self.param(
            _name("linear", i),
            ...,
            (16384, 2048)
        )
        outputs = jnp.dot(activations, w_linear)  # [B, T, 2048]

        out.append(outputs)
        gates.append(gate)

# ─── Second Gated Residual ───
xs = [
    _gated_residual(xs[0], out[0], gates[0]),
    _gated_residual(xs[1], out[1], gates[1]),
]

# ✅ Output (Layer 0 완료):
# xs[0]: [4, 784, 2048]  (Prefix after full transformer block)
# xs[1]: [4, 32, 2048]   (Suffix after full transformer block)
```

---

## 📍 Step 8: Transformer Layers 1-17

**코드 위치**: `src/openpi/models/gemma.py:365-381` (nn.scan)

```python
# ═══════════════════════════════════════════════════════════════
# Step 8: Repeat Layer 0 for Layers 1-17
# ═══════════════════════════════════════════════════════════════

# nn.scan을 사용하여 18개 layer를 자동으로 반복
# 각 layer는 동일한 구조, 다른 weight

for layer_idx in range(1, 18):
    # Layer 0과 동일한 과정 반복:
    # 1. Pre-Attention RMSNorm (with AdaRMS)
    # 2. QKV Projection (Expert-specific)
    # 3. RoPE
    # 4. Grouped Query Attention
    # 5. Output Projection (Expert-specific)
    # 6. Gated Residual
    # 7. Pre-FFN RMSNorm (with AdaRMS)
    # 8. FeedForward (Expert-specific)
    # 9. Gated Residual

    pass  # Automatically handled by nn.scan

# ✅ After 18 layers:
# xs[0]: [4, 784, 2048]  (Prefix, fully processed)
# xs[1]: [4, 32, 2048]   (Suffix, fully processed)
```

---

## 📍 Step 9: Final Layer Normalization

**코드 위치**: `src/openpi/models/gemma.py:409-411`

```python
# ═══════════════════════════════════════════════════════════════
# Step 9: Final RMSNorm (Expert별)
# ═══════════════════════════════════════════════════════════════

# xs[0]: [4, 784, 2048]
# xs[1]: [4, 32, 2048]

outputs = []
for i, (x, final_norm) in enumerate(zip(xs, self.final_norms)):
    if x is not None:
        # Final RMSNorm (no AdaRMS here)
        x_final, _ = final_norm(x, adarms_cond[i])
        outputs.append(x_final)
    else:
        outputs.append(None)

# ✅ Output:
# outputs[0]: [4, 784, 2048]  (Prefix final output)
# outputs[1]: [4, 32, 2048]   (Suffix final output)
```

---

## 📍 Step 10: Velocity Prediction

**코드 위치**: `src/openpi/models/pi0.py:212`

```python
# ═══════════════════════════════════════════════════════════════
# Step 10: Action Tokens → Velocity Prediction
# ═══════════════════════════════════════════════════════════════

# suffix_out: [4, 32, 2048]

# Only take the last action_horizon tokens
action_output = suffix_out[:, -32:]  # [4, 32, 2048]

# Project to action dimension
v_t = self.action_out_proj(action_output)
# Linear(2048 → 7)
# v_t: [4, 32, 7]

# ✅ Output:
# v_t: [4, 32, 7]  (Predicted velocity field)
```

---

## 📍 Step 11: Flow Matching Loss

**코드 위치**: `src/openpi/models/pi0.py:214`

```python
# ═══════════════════════════════════════════════════════════════
# Step 11: Compute Flow Matching Loss
# ═══════════════════════════════════════════════════════════════

# ─── Target velocity ───
u_t = noise - actions
# [4, 32, 7] - [4, 32, 7] = [4, 32, 7]

# ─── Loss ───
loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
# MSE loss averaged over action dimensions
# loss: [4, 32]  (loss per timestep)

# Average over timesteps
final_loss = jnp.mean(loss)
# final_loss: scalar

# ✅ Output:
# final_loss: scalar (training objective)
```

---

## 📍 Inference: Flow Matching Sampling

**코드 위치**: `src/openpi/models/pi0.py:217-279`

Inference 시에는 noise에서 시작해서 점진적으로 denoising합니다.

```python
# ═══════════════════════════════════════════════════════════════
# Inference: Iterative Denoising (Euler Integration)
# ═══════════════════════════════════════════════════════════════

def sample_actions(self, rng, observation, num_steps=10):
    # ─── Step I-1: Prefix KV Cache 생성 (한 번만) ───
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],  # Expert 0만 처리
        mask=prefix_attn_mask,
        positions=positions,
    )
    # kv_cache: [18, 4, 784, 1, 256]  ← 저장!

    # ─── Step I-2: 초기화 ───
    noise = jax.random.normal(rng, (4, 32, 7))
    x_t = noise  # time=1.0에서 시작 (pure noise)
    dt = -1.0 / num_steps  # -0.1

    # ─── Step I-3: Iterative Denoising ───
    def step(carry):
        x_t, time = carry
        # time: 1.0 → 0.9 → 0.8 → ... → 0.1 → 0.0

        # Suffix embedding
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
            self.embed_suffix(observation, x_t, jnp.broadcast_to(time, [4]))

        # Attention mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=32)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

        # Positions
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Transformer (Expert 1만, KV cache 재사용!)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # Prefix는 None (cache에서 가져옴)
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,  # ← 저장된 cache 재사용!
            adarms_cond=[None, adarms_cond],
        )

        # Velocity 예측
        v_t = self.action_out_proj(suffix_out[:, -32:])

        # Euler integration: x_{t+dt} = x_t + dt * v_t
        x_t = x_t + dt * v_t
        time = time + dt

        return x_t, time

    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2  # time > 0

    # While loop (10 iterations)
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

    return x_0  # [4, 32, 7]  ← Denoised actions!

# ✅ Inference 결과:
# x_0: [4, 32, 7]  (Clean actions)
```

---

## 📊 전체 데이터 흐름 요약

### Shape 변화 추적

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Data                             │
├─────────────────────────────────────────────────────────────┤
│ Images:  3 × [4, 224, 224, 3]  uint8 [0, 255]             │
│ State:   [4, 7]                float32                     │
│ Text:    [4, 16]               int32                       │
│ Actions: [4, 32, 7]            float32 (training)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Step 1: Preprocessing                    │
├─────────────────────────────────────────────────────────────┤
│ Images → float32 [-1, 1]                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Step 2-4: Prefix Embedding                     │
├─────────────────────────────────────────────────────────────┤
│ Images:  3 × [4, 256, 2048]    (SigLIP)                   │
│ Text:    [4, 16, 2048]         (Embedder)                 │
│ ──────────────────────────────────────────                 │
│ Prefix:  [4, 784, 2048]        (Concatenated)             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               Step 5: Suffix Embedding                      │
├─────────────────────────────────────────────────────────────┤
│ Actions: [4, 32, 7] → [4, 32, 2048]  (Linear projection)  │
│ Time:    [4] → [4, 2048]             (Sinusoidal + MLP)   │
│ ──────────────────────────────────────────                 │
│ Suffix:  [4, 32, 2048]                                     │
│ AdaRMS:  [4, 2048]                   (conditioning)        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 6: Attention Mask Generation                  │
├─────────────────────────────────────────────────────────────┤
│ Mask:     [4, 816, 816]         (Prefix-LM + Causal)      │
│ Positions: [4, 816]                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│        Step 7-8: Multi-Expert Transformer (18 layers)       │
├─────────────────────────────────────────────────────────────┤
│ xs[0]: [4, 784, 2048] ──→ ... ──→ [4, 784, 2048]          │
│        (Expert 0 weights)                                   │
│                                                             │
│ xs[1]: [4, 32, 2048]  ──→ ... ──→ [4, 32, 2048]           │
│        (Expert 1 weights, AdaRMS conditioning)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│             Step 9: Final Normalization                     │
├─────────────────────────────────────────────────────────────┤
│ Prefix:  [4, 784, 2048]                                    │
│ Suffix:  [4, 32, 2048]                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 10: Velocity Prediction                       │
├─────────────────────────────────────────────────────────────┤
│ v_t: [4, 32, 7]                 (Predicted velocity)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│           Step 11: Loss Computation                         │
├─────────────────────────────────────────────────────────────┤
│ loss = mean_squared_error(v_t, u_t)                        │
│      = mean((v_t - (noise - actions))^2)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔥 핵심 포인트

### 1. Multi-Expert 메커니즘

```python
# 각 expert는 자신만의 weight 사용:

# Expert 0 (Prefix - Image/Text):
# - "q_einsum": [8, 2048, 256]
# - "kv_einsum": [2, 1, 2048, 256]
# - "attn_vec_einsum": [8, 256, 2048]
# - "mlp/gating_einsum": [2, 2048, 16384]
# - "mlp/linear": [16384, 2048]

# Expert 1 (Suffix - Action):
# - "q_einsum_1": [8, 2048, 256]     ← 다른 weight!
# - "kv_einsum_1": [2, 1, 2048, 256]
# - "attn_vec_einsum_1": [8, 256, 2048]
# - "mlp/gating_einsum_1": [2, 2048, 16384]
# - "mlp/linear_1": [16384, 2048]
```

### 2. AdaRMS Conditioning (π₀.₅)

```python
# Timestep을 adaptive normalization으로 주입:

time_emb = posemb_sincos(time)  # [4, 2048]

# RMSNorm에서:
modulation = Dense(2048 * 3)(time_emb)  # [4, 6144]
scale, shift, gate = split(modulation, 3)  # 각 [4, 2048]

normed = normed * (1 + scale[:, None, :]) + shift[:, None, :]
# Timestep에 따라 normalization 파라미터 변경!

# Gated residual:
x = x + y * gate[:, None, :]
# Timestep에 따라 residual 강도 조절!
```

### 3. Flow Matching

```python
# Training:
time ~ Beta(1.5, 1.0)  # [0, 1]
x_t = time * noise + (1 - time) * actions
u_t = noise - actions  # Target velocity
loss = ||v_t - u_t||^2

# Inference (Euler integration):
x_t = noise  # t=1
for t in [1.0, 0.9, 0.8, ..., 0.1, 0.0]:
    v_t = model(x_t, t)
    x_t = x_t - 0.1 * v_t  # Euler step
# x_0 = clean actions
```

### 4. Attention Pattern

```
Prefix (Image/Text):
  ├─ Bidirectional attention
  └─ Can attend to everything

Suffix (Action):
  ├─ Can attend to Prefix
  └─ Causal attention within Suffix
```

### 5. KV Cache Reuse (Inference)

```python
# Prefix를 한 번만 처리:
_, kv_cache = llm([prefix_tokens, None], ...)
# kv_cache: [18 layers, 4, 784, 1, 256]

# 10번 반복할 때마다 재사용:
for step in range(10):
    _, _ = llm([None, suffix_tokens], kv_cache=kv_cache, ...)
    # Prefix는 재계산 안 함! ← 10배 빠름
```

---

## 📝 변경 이력

- 2026-02-08: 초안 작성
  - 전체 데이터 흐름 Step-by-Step 정리
  - 각 단계별 상세 코드 설명
  - Shape 변화 추적
  - 핵심 포인트 정리

---

**작성자**: AI Analysis
**프로젝트**: openpi (Physical Intelligence)
**버전**: 1.0
