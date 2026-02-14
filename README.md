# π₀ 모델 데이터 흐름 Step-by-Step 완전 가이드  
  
이 문서는 π₀ 모델에서 **입력 데이터가 어떻게 처리되어 최종 출력이 되는지**를 한 단계씩 추적합니다.  
  
> **📌 학습 vs 추론 구분**  
> 이 문서는 **학습(Training)** 과정을 기본으로 설명하며, 각 Step에서 추론   (Inference)과 차이가 있는 경우 `🔄 학습 vs 추론` 박스로 표시합니다.  
> - 🏋️ **학습**: Ground truth actions + noise → Flow Matching loss 계산   
> - 🎯 **추론**: Pure noise에서 시작 → 10회 Euler integration으로 action 생성  
  
Step 0: 원본 입력 데이터 (Images, State, Text, Actions)    
Step 1: Observation 객체 생성 (uint8 → float32 정규화)  
Step 2: Image Embedding (SigLIP) - 3×256 = 768 tokens  
Step 3: Text Embedding (Gemma Embedder) - 16 tokens  
Step 4: Prefix Concatenation - 784 tokens (Image + Text)  
Step 5: Action Embedding (Suffix) - 33 tokens (State 1 + Action 32) + Flow Matching  
Step 6: Attention Mask 생성 - [4, 817, 817]  
Step 7: Transformer Layer 0 상세 분석  
7-1: Pre-Attention RMSNorm  
7-2: QKV Projection (Multi-Expert)  
7-3: RoPE (Rotary Position Embedding)  
7-4: Grouped Query Attention  
7-5: Output Projection (Expert별)  
7-6: Residual Connection  
7-7: FeedForward Network  
Step 8: Transformer Layers 1-17 (18 layers total)  
Step 9: Final Layer Normalization  
Step 10-11: Velocity Prediction + Flow Matching Loss  
    

**예시 데이터**:  
- Batch Size: B = 4  
- Images: 3개 (base_0, left_wrist_0, right_wrist_0)  
- Text: 16 tokens  
- Actions: 32 timesteps, 7 DoF  
- Model: π₀  
  
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 0 요약:
모델에 들어가는 원본 입력 데이터의 구조.
  - Images: 3대의 카메라 (base, left_wrist, right_wrist)에서 찍은 RGB 영상
  - State:  로봇의 현재 관절 상태 (x, y, z, 쿼터니언, 그리퍼) 7 DoF
  - Text:   사람이 내린 언어 명령 ("pick up fork" 등) → 이미 토크나이즈된 정수 배열
  - Actions: [학습 전용] 전문가가 수행한 Ground truth 행동 시퀀스
             추론 시에는 없음 → noise 에서 생성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 1 요약:
  딕셔너리 형태의 원본 데이터를 구조화된 Observation 객체로 변환.
  - uint8 [0,255] 이미지 → float32 [-1,1] 로 정규화
    (모델이 연속적인 실수값 입력을 기대하기 때문)
  - 이후 모든 처리는 이 Observation 객체를 통해 접근
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

    # 2-3. Transformer Encoder (27 layers, So400m variant)
    for layer in range(27):
        # Pre-Norm
        x_norm = nn.LayerNorm()(x)

        # Multi-Head Self-Attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=16,  # 1152 / 16 = 72 per head
        )(x_norm, x_norm)  # [4, 256, 1152]

        x = x + attn_out  # Residual

        # Pre-Norm
        x_norm = nn.LayerNorm()(x)

        # MLP
        mlp_out = nn.Dense(4304)(x_norm)  # [4, 256, 4304]
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(1152)(mlp_out)  # [4, 256, 1152]

        x = x + mlp_out  # Residual

    # 2-4. Final Projection to PaliGemma dimension
    image_tokens = nn.Dense(2048)(x)  # [4, 256, 2048]

    image_tokens_list.append(image_tokens)

# ✅ Output:
# image_tokens_list = [
#     [4, 256, 2048],  # base_0_rgb
#     [4, 256, 2048],  # left_wrist_0_rgb
#     [4, 256, 2048],  # right_wrist_0_rgb
# ]
# Total: 3 × 256 = 768 image tokens

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 2 요약:
  SigLIP (ViT-So400m/14) 으로 이미지를 패치 토큰 시퀀스로 변환.
  - 224×224 이미지를 14×14 크기의 패치 256개로 분할
  - 27층 ViT Transformer로 각 패치의 문맥적 특징 추출 (width=1152)
  - 최종 Dense(2048) 로 PaliGemma 의 언어 모델 차원에 맞게 투영
  - 3개 카메라 각각 독립 처리 → 768개의 이미지 토큰 생성
  - 이 토큰들이 언어 토큰과 동일한 임베딩 공간에 놓이게 됨
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 3 요약:
  정수 토큰 ID를 연속적인 2048차원 임베딩 벡터로 변환.
  - 어휘 크기 257,152개의 룩업 테이블에서 해당 행을 가져옴
  - √2048 ≈ 45.25 로 스케일링하여 임베딩 크기를 안정화
    (이미지 토큰과 언어 토큰이 같은 수치 범위에 있도록 맞춤)
  - 이 시점에서 이미지 토큰과 텍스트 토큰은 동일한 [B, S, 2048] 형태를 가짐
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 4 요약:
  이미지 토큰(768개)과 텍스트 토큰(16개)을 하나의 Prefix 시퀀스로 합침.
  - 순서: [image_0(256), image_1(256), image_2(256), text(16)] = 784 토큰
  - ar_mask = 전부 False → Prefix 내부는 모든 토큰이 서로를 볼 수 있는 양방향 attention
  - 이 Prefix는 "환경 관찰 정보" 전체를 담음
  - 추론 시 이 784개 토큰은 KV Cache로 저장되어 한 번만 계산됨
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 5: Action Embedding (Suffix)

**코드 위치**: `src/openpi/models/pi0.py:139-186`

```python
# ═══════════════════════════════════════════════════════════════
# Step 5: State + Actions → Suffix Tokens (π₀ 방식)
# ═══════════════════════════════════════════════════════════════

# Training input: ground truth actions
actions = raw_input["actions"]  # [4, 32, 7]
state = observation.state       # [4, 7]

# ─── 5-1: Flow Matching Preparation ───
noise = jax.random.normal(noise_rng, actions.shape)  # [4, 32, 7]

# Timestep 샘플링 (Beta distribution, t=1이 noise, t=0이 data)
time = jax.random.beta(time_rng, 1.5, 1.0, batch_shape=[4])
# time: [4]  예: [0.234, 0.891, 0.456, 0.123]

# Flow interpolation: x_t = t·noise + (1-t)·actions
time_expanded = time[:, None, None]  # [4, 1, 1]
x_t = time_expanded * noise + (1 - time_expanded) * actions
# x_t: [4, 32, 7]

# Target velocity (직선 경로이므로 상수)
u_t = noise - actions  # [4, 32, 7]

# ─── 5-2: State Token (π₀ 전용) ───
# state를 1개의 토큰으로 projection
state_token = self.state_proj(state)[:, None, :]
# Linear(7 → 1024)
# [4, 7] → [4, 1024] → [4, 1, 1024]

# ─── 5-3: Action Token Projection ───
action_tokens = self.action_in_proj(x_t)
# Linear(7 → 1024)
# [4, 32, 7] → [4, 32, 1024]

# ─── 5-4: Timestep Embedding (Sinusoidal) ───
def posemb_sincos(pos, embedding_dim, min_period, max_period):
    # pos: [4], embedding_dim: 1024
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # [512]
    period = min_period * (max_period / min_period) ** fraction
    # period: [0.004, ..., 4.0]

    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,            # [4]
        1.0 / period * 2 * jnp.pi  # [512]
    )  # [4, 512]

    emb = jnp.concatenate([
        jnp.sin(sinusoid_input),  # [4, 512]
        jnp.cos(sinusoid_input),  # [4, 512]
    ], axis=-1)  # [4, 1024]

    return emb

time_emb = posemb_sincos(time, 1024, min_period=4e-3, max_period=4.0)
# time_emb: [4, 1024]

# ─── 5-5: Action + Time 결합 MLP (π₀ 방식) ───
# timestep 임베딩을 action_horizon만큼 복제
time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=32)
# [4, 1024] → [4, 32, 1024]

# action과 time을 concat
action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
# [4, 32, 1024] + [4, 32, 1024] = [4, 32, 2048]

# MLP로 압축
action_time_tokens = self.action_time_mlp_in(action_time_tokens)
# Linear(2048 → 1024): [4, 32, 2048] → [4, 32, 1024]
action_time_tokens = nnx.swish(action_time_tokens)
action_time_tokens = self.action_time_mlp_out(action_time_tokens)
# Linear(1024 → 1024): [4, 32, 1024] → [4, 32, 1024]
action_expert_tokens = action_time_tokens  # [4, 32, 1024]

# ─── 5-6: Suffix 구성 ───
# state token + action tokens concat
suffix_tokens = jnp.concatenate([state_token, action_expert_tokens], axis=1)
# [4, 1, 1024] + [4, 32, 1024] = [4, 33, 1024]

suffix_mask = jnp.ones([4, 33], dtype=bool)  # 모두 valid

# AR Mask:
# - state 토큰: [True]          ← prefix가 state를 볼 수 없음
# - 첫 action 토큰: [True]      ← state가 action을 볼 수 없음
# - 나머지 action 토큰: [False×31] ← action끼리 양방향 attention
suffix_ar_mask = jnp.array([True] + [True] + [False] * 31)
# [True, True, False, False, ..., False]
#  state  act0  act1  ...        act31

# ✅ Output:
# - suffix_tokens: [4, 33, 1024]
#   ├─ state:   tokens[0]     ← 1개
#   └─ actions: tokens[1:33]  ← 32개
# - suffix_mask: [4, 33] (all True)
# - suffix_ar_mask: [33] ([True, True, False×31])
# - adarms_cond: None (π₀는 AdaRMS 사용 안 함)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 5 요약:
  로봇 state와 (노이즈 섞인) action을 Action Expert 차원(1024)으로 임베딩.
  - State:  Linear(7→1024) → 1개의 state 토큰
  - Action: Linear(7→1024) → 32개의 action 토큰
  - Time:   sincos PE로 스칼라 t → 1024차원 벡터로 변환
            action 토큰과 concat 후 MLP → 시간 정보를 action 임베딩에 혼합
  - Flow Matching: x_t = t·noise + (1-t)·actions  (학습 시 중간 상태 생성)
  - suffix = [state(1), action(32)] = 33 토큰
  - ar_mask: [True, True, False×31]
    → state(cumsum=1)는 action(cumsum=2)을 볼 수 없음
    → action끼리는 양방향 attention (cumsum=2 끼리 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# [4, 784] + [4, 33] = [4, 817]

ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
# [784] + [33] = [817]
# ar_mask = [False×784, True, True, False×31]
#            ^^^^^^^^    ^     ^     ^^^^^^^^
#            Prefix      state act0  act1~31

# ─── 6-2: cumsum으로 그룹 분리 ───
cumsum = jnp.cumsum(ar_mask, axis=1)
# [0×784, 1, 2, 2×31]
#  Prefix  st act0 act1~31
# → 그룹 0: prefix
# → 그룹 1: state
# → 그룹 2: action (모두 동일, 양방향)

# ─── 6-3: Generate Attention Mask ───
def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # [4, 817]
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # cumsum[key] <= cumsum[query] 이면 query가 key를 볼 수 있음
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

attn_mask = make_attn_mask(input_mask, ar_mask)
# attn_mask: [4, 817, 817]

# ─── 6-4: Attention 패턴 시각화 ───
"""
Attention pattern [817, 817]:

                  Prefix(784)        State(1)  Actions(32)
              cumsum=0               cumsum=1  cumsum=2
         ┌──────────────────────┬──────────┬────────────┐
Prefix   │  ✓  ✓  ✓  ...  ✓   │    ✗     │  ✗  ...  ✗ │ cumsum=0
(0-783)  │  (양방향)             │          │            │
         ├──────────────────────┼──────────┼────────────┤
State    │  ✓  ✓  ✓  ...  ✓   │    ✓     │  ✗  ...  ✗ │ cumsum=1
(784)    │  prefix 참조 가능     │  자기자신  │            │
         ├──────────────────────┼──────────┼────────────┤
Actions  │  ✓  ✓  ✓  ...  ✓   │    ✓     │  ✓  ...  ✓ │ cumsum=2
(785-816)│  prefix 참조 가능     │  state참조│  (양방향)  │
         └──────────────────────┴──────────┴────────────┘

규칙: cumsum[key] <= cumsum[query] → 참조 가능
  prefix→prefix:   0<=0 ✓ 양방향
  prefix→state:    1<=0 ✗ 차단
  prefix→action:   2<=0 ✗ 차단
  state→prefix:    0<=1 ✓
  state→state:     1<=1 ✓
  state→action:    2<=1 ✗ 차단
  action→prefix:   0<=2 ✓
  action→state:    1<=2 ✓
  action→action:   2<=2 ✓ 양방향
"""

# ─── 6-5: Position Encoding ───
positions = jnp.cumsum(input_mask, axis=1) - 1
# [4, 817]
# Example: [[0, 1, 2, ..., 783, 784, 785, ..., 816], ...]

# ✅ Output:
# - attn_mask: [4, 817, 817] bool
# - positions: [4, 817] int32

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 6 요약:
  어떤 토큰이 어떤 토큰을 볼 수 있는지 결정하는 [817,817] 마스크 생성.
  - ar_mask를 cumsum으로 그룹화: prefix(0) / state(1) / action(2)
  - 규칙: cumsum[key] <= cumsum[query] 이면 참조 가능
    → prefix끼리 양방향  (0<=0)
    → action→prefix 가능 (0<=2), prefix→action 불가 (2<=0 ✗)
    → action→state 가능  (1<=2), state→action 불가  (2<=1 ✗)
    → action끼리 양방향  (2<=2)
  - 이 설계의 핵심: prefix는 suffix에 영향받지 않음
    → 추론 시 prefix KV Cache를 안전하게 재사용할 수 있는 근거
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 7: Multi-Expert Transformer Layer 0

이제 18개의 Transformer layer 중 **첫 번째 layer**를 자세히 봅니다.

**코드 위치**: `src/openpi/models/gemma.py:284-333`

### Step 7-1: Pre-Attention RMSNorm

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-1: Pre-Attention RMSNorm
# ═══════════════════════════════════════════════════════════════

# 입력:
xs = [prefix_tokens, suffix_tokens]
# xs[0]: [4, 784, 2048]  (Expert 0 - PaliGemma, width=2048)
# xs[1]: [4, 33, 1024]   (Expert 1 - Action Expert, width=1024)

adarms_cond = [None, None]
# π₀는 AdaRMS 사용 안 함 → 둘 다 None

# ─── RMSNorm 적용 (두 expert 모두 동일한 방식) ───
pre_attn = []
gates = []

for i, x in enumerate(xs):
    if x is not None:
        # 1. Root Mean Square 계산
        var = jnp.mean(jnp.square(x.astype(float32)), axis=-1, keepdims=True)
        # xs[0]: [4, 784, 2048] → var: [4, 784, 1]
        # xs[1]: [4, 33, 1024]  → var: [4, 33, 1]

        # 2. Normalization
        normed_inputs = x * jnp.reciprocal(jnp.sqrt(var + 1e-6))

        # 3. Regular RMSNorm (두 expert 모두 동일)
        scale = self.param("scale", zeros_init(), (x.shape[-1],))
        # Expert 0: scale [2048]
        # Expert 1: scale [1024]
        x_norm = normed_inputs * (1 + scale)
        gate = None  # π₀는 gate 없음

        pre_attn.append(x_norm)
        gates.append(gate)

# ✅ Output:
# pre_attn[0]: [4, 784, 2048]  (Normalized prefix)
# pre_attn[1]: [4, 33, 1024]   (Normalized suffix)
# gates = [None, None]          (π₀는 gate 없음)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-1 요약:
  Attention 전에 각 expert의 입력을 독립적으로 정규화.
  - Expert 0 (PaliGemma): scale 파라미터 크기 [2048]
  - Expert 1 (Action Expert): scale 파라미터 크기 [1024]
  - 두 expert 모두 일반 RMSNorm 사용 (π₀는 AdaRMS 없음)
  - RMSNorm: 각 토큰 벡터의 RMS로 나누어 크기를 맞춤
    (LayerNorm과 달리 평균 빼기 없이 분산만 정규화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-2: QKV Projection (Multi-Expert)

**코드 위치**: `src/openpi/models/gemma.py:158-199`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-2: QKV Projection (Expert-specific Weights)
# ═══════════════════════════════════════════════════════════════

# head_dim=256, num_heads=8, num_kv_heads=1 은 두 expert 공통

qkvs = []

# ─── Expert 0 (PaliGemma, width=2048) ───
q_einsum_0 = Einsum(shape=(8, 2048, 256), name="q_einsum")
q_0 = q_einsum_0("BTD,NDH->BTNH", pre_attn[0])
# [4, 784, 2048] → q_0: [4, 784, 8, 256]

kv_einsum_0 = Einsum(shape=(2, 1, 2048, 256), name="kv_einsum")
k_0, v_0 = kv_einsum_0("BSD,2KDH->2BSKH", pre_attn[0])
# k_0, v_0: [4, 784, 1, 256]

qkvs.append((q_0, k_0, v_0))

# ─── Expert 1 (Action Expert, width=1024) ───
q_einsum_1 = Einsum(shape=(8, 1024, 256), name="q_einsum_1")  # ← width=1024!
q_1 = q_einsum_1("BTD,NDH->BTNH", pre_attn[1])
# [4, 33, 1024] → q_1: [4, 33, 8, 256]

kv_einsum_1 = Einsum(shape=(2, 1, 1024, 256), name="kv_einsum_1")  # ← width=1024!
k_1, v_1 = kv_einsum_1("BSD,2KDH->2BSKH", pre_attn[1])
# k_1, v_1: [4, 33, 1, 256]

qkvs.append((q_1, k_1, v_1))

# ✅ Output:
# qkvs[0]: (q[4,784,8,256], k[4,784,1,256], v[4,784,1,256]) ← 2048→256
# qkvs[1]: (q[4,33,8,256],  k[4,33,1,256],  v[4,33,1,256])  ← 1024→256
#          ^^^ 입력 차원은 다르지만, 출력(head_dim=256)은 같음!

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-2 요약:
  각 expert가 서로 다른 가중치로 QKV를 계산하되, 출력 head_dim=256은 통일.
  - Expert 0: 2048 → 256 (Q: 8heads, K/V: 1head)
  - Expert 1: 1024 → 256 (Q: 8heads, K/V: 1head)
  - 다른 입력 차원을 같은 attention 공간으로 매핑하는 핵심 단계
  - K/V head 수=1 (Grouped Query Attention): 메모리 절약
    → 8개 Q가 1개 K,V를 공유 → 8배 메모리 절약
  - 이 projection 후 Q,K,V를 concat하여 shared attention 계산 가능
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-3: RoPE (Rotary Position Embedding)

**코드 위치**: `src/openpi/models/gemma.py:424-440`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-3: Apply RoPE to Q and K
# ═══════════════════════════════════════════════════════════════

# 두 expert의 QKV를 sequence 축으로 concat → 같은 256 차원으로 합쳐짐
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs))
# q: [4, 817, 8, 256]  (784 + 33)
# k: [4, 817, 1, 256]
# v: [4, 817, 1, 256]

def _apply_rope(x, positions, max_wavelength=10_000):
    freq_exponents = (2.0 / 256) * jnp.arange(256 // 2)  # [128]
    timescale = max_wavelength ** freq_exponents

    radians = positions[..., None] / timescale[None, None, :]
    # [4, 817, 128]
    radians = radians[..., None, :]  # [4, 817, 1, 128]

    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)

    res = jnp.concatenate([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], axis=-1)  # [4, 817, H, 256]

    return res

q = _apply_rope(q, positions=positions)  # [4, 817, 8, 256]
k = _apply_rope(k, positions=positions)  # [4, 817, 1, 256]
q *= 256 ** -0.5  # scale by 1/√head_dim

# ✅ Output:
# q: [4, 817, 8, 256]
# k: [4, 817, 1, 256]
# v: [4, 817, 1, 256]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-3 요약:
  두 expert의 QKV를 시퀀스 축으로 concat 후 위치 인코딩 적용.
  - concat: prefix(784) + suffix(33) = 817 토큰으로 합쳐짐
    → 이 시점부터 두 expert의 토큰이 하나의 시퀀스로 처리됨 (Shared Attention)
  - RoPE: 절대 위치 인코딩과 달리 Q,K에만 회전 변환을 적용
    → 토큰 간 상대 위치가 내적(attention score)에 자연스럽게 반영됨
  - q에 1/√256 스케일링: softmax 전 값이 너무 커지지 않도록 안정화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-4: Grouped Query Attention

**코드 위치**: `src/openpi/models/gemma.py:216-231`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-4: Grouped Query Attention (GQA)
# ═══════════════════════════════════════════════════════════════

# Reshape Q for GQA: num_heads=8, num_kv_heads=1, group_size=8
q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=1)
# [4, 817, 8, 256] → [4, 817, 1, 8, 256]

# Attention scores
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
# q: [4, 817, 1, 8, 256]
# k: [4, 817, 1, 256]
# logits: [4, 1, 8, 817, 817]

# Apply attention mask
big_neg = -2.3819763e38
attn_mask_expanded = attn_mask[:, None, None, :, :]
# [4, 817, 817] → [4, 1, 1, 817, 817]

masked_logits = jnp.where(attn_mask_expanded, logits, big_neg)

probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
# probs: [4, 1, 8, 817, 817]

encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
# encoded: [4, 817, 1, 8, 256]

encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
# [4, 817, 8, 256]

# ✅ Output:
# encoded: [4, 817, 8, 256]  (attention-weighted values)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-4 요약:
  817개 전체 토큰(prefix+suffix)에 대해 attention 계산.
  - Step 6에서 만든 [817,817] 마스크를 적용
    → 허용되지 않은 위치는 -∞로 설정 → softmax 후 확률 0
  - 두 expert의 토큰이 하나의 attention 행렬을 공유
    → Action Expert 토큰이 PaliGemma 토큰(이미지/언어)을 직접 참조 가능
  - 이것이 Transfusion 구조의 핵심:
    서로 다른 모달리티(언어, 이미지, 행동)가 하나의 attention에서 상호작용
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-5: Output Projection (Multi-Expert)

**코드 위치**: `src/openpi/models/gemma.py:233-249`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-5: Output Projection (Expert별 독립)
# ═══════════════════════════════════════════════════════════════

# encoded: [4, 817, 8, 256] (모든 토큰의 attention output)

out = []
start = 0

# ─── Expert 0 (PaliGemma) ───
end = 784
expert_encoded_0 = encoded[:, start:end]  # [4, 784, 8, 256]
out_einsum_0 = Einsum(
    shape=(8, 256, 2048),  # (num_heads, head_dim, width)
    name="attn_vec_einsum"
)
expert_out_0 = out_einsum_0("BTNH,NHD->BTD", expert_encoded_0)
# [4, 784, 2048]
out.append(expert_out_0)
start = end  # 784

# ─── Expert 1 (Action Expert) ───
end = 817
expert_encoded_1 = encoded[:, start:end]  # [4, 33, 8, 256]
out_einsum_1 = Einsum(
    shape=(8, 256, 1024),  # ← width=1024!
    name="attn_vec_einsum_1"
)
expert_out_1 = out_einsum_1("BTNH,NHD->BTD", expert_encoded_1)
# [4, 33, 1024]
out.append(expert_out_1)

# ✅ Output:
# out[0]: [4, 784, 2048]  (Prefix, Expert 0 weight 사용)
# out[1]: [4, 33, 1024]   (Suffix, Expert 1 weight 사용)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-5 요약:
  공유 attention 결과를 다시 각 expert의 차원으로 분리하여 복원.
  - encoded [4,817,8,256]를 앞 784개/뒤 33개로 분할
  - Expert 0: 8×256 → 2048  (PaliGemma 원래 차원 복원)
  - Expert 1: 8×256 → 1024  (Action Expert 원래 차원 복원)
  - 각 expert가 서로 다른 출력 projection 가중치를 가짐
  - 이로써 공유 attention 정보가 각자의 표현 공간으로 매핑됨
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-6: Residual Connection

**코드 위치**: `src/openpi/models/gemma.py:309-312` + `453-459`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-6: Residual Connection (π₀는 gate 없음)
# ═══════════════════════════════════════════════════════════════

def _gated_residual(x, y, gate):
    if gate is None:
        return x + y  # π₀: 일반 residual
    return x + y * gate  # π₀.₅: gated residual (미사용)

xs = [
    _gated_residual(xs[0], out[0], gates[0]),  # gate=None → 단순 합
    _gated_residual(xs[1], out[1], gates[1]),  # gate=None → 단순 합
]

# Expert 0 (Prefix):
# xs[0] = prefix_tokens + out[0]
# [4, 784, 2048] + [4, 784, 2048] = [4, 784, 2048]

# Expert 1 (Suffix):
# xs[1] = suffix_tokens + out[1]
# [4, 33, 1024] + [4, 33, 1024] = [4, 33, 1024]

# ✅ Output:
# xs[0]: [4, 784, 2048]
# xs[1]: [4, 33, 1024]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-6 요약:
  Attention 출력을 원래 입력에 더하는 첫 번째 Residual Connection.
  - xs[i] = xs[i] + attn_out[i]  (원래 정보 + attention으로 얻은 새 정보)
  - Residual의 역할: 깊은 네트워크에서 기울기 소실 방지
    → attention이 0에 가까워도 원래 신호가 그대로 흐름
  - π₀는 gate=None → 단순 덧셈 (π₀.₅는 gate로 가중 합산)
  - 두 expert 각각 독립적으로 수행 (차원 유지: 2048, 1024)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-7: FeedForward Network

**코드 위치**: `src/openpi/models/gemma.py:314-330`

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-7: FeedForward Network (Expert별 독립)
# ═══════════════════════════════════════════════════════════════

for i, (x, config) in enumerate(zip(xs, configs)):
    if x is not None:
        # ─── Pre-FFN RMSNorm ───
        x_norm, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, None)
        # gate = None (π₀)

        # ─── GeGLU FeedForward ───
        if i == 0:  # Expert 0 (PaliGemma)
            # width=2048, mlp_dim=16384
            w_gating = param((2, 2048, 16384))
            ff_gate = jnp.dot(x_norm, w_gating[0])   # [4, 784, 16384]
            ff1     = jnp.dot(x_norm, w_gating[1])   # [4, 784, 16384]
            activations = nn.gelu(ff_gate) * ff1       # [4, 784, 16384]
            w_linear = param((16384, 2048))
            outputs = jnp.dot(activations, w_linear)   # [4, 784, 2048]

        else:       # Expert 1 (Action Expert)
            # width=1024, mlp_dim=4096
            w_gating = param((2, 1024, 4096))
            ff_gate = jnp.dot(x_norm, w_gating[0])   # [4, 33, 4096]
            ff1     = jnp.dot(x_norm, w_gating[1])   # [4, 33, 4096]
            activations = nn.gelu(ff_gate) * ff1       # [4, 33, 4096]
            w_linear = param((4096, 1024))
            outputs = jnp.dot(activations, w_linear)   # [4, 33, 1024]

# ─── Second Residual ───
xs = [
    _gated_residual(xs[0], out[0], None),  # [4, 784, 2048]
    _gated_residual(xs[1], out[1], None),  # [4, 33, 1024]
]

# ✅ Output (Layer 0 완료):
# xs[0]: [4, 784, 2048]  (Prefix after full transformer block)
# xs[1]: [4, 33, 1024]   (Suffix after full transformer block)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-7 요약:
  Attention 이후 각 토큰을 독립적으로 비선형 변환하는 FFN.
  - Expert별 완전히 독립적인 가중치 사용
  - Expert 0: 2048 → 16384 → 2048  (8배 확장 후 복원)
  - Expert 1: 1024 → 4096 → 1024   (4배 확장 후 복원)
  - GeGLU 활성화 (GELU gate × linear): 정보 선택적 통과
  - 두 번째 Residual: xs[i] = xs[i] + ffn_out[i]
  - FFN이 attention이 섞어온 정보를 각 expert의 "개인 처리"로 소화
  - Layer 0 완료 → Layer 1~17도 동일 과정 반복
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    # 1. Pre-Attention RMSNorm (Expert별 독립 scale)
    # 2. QKV Projection (Expert 0: 2048→256, Expert 1: 1024→256)
    # 3. Q, K concat → RoPE 적용
    # 4. Grouped Query Attention (shared)
    # 5. Output Projection (Expert 0: 256→2048, Expert 1: 256→1024)
    # 6. Residual
    # 7. Pre-FFN RMSNorm
    # 8. FeedForward (Expert 0: 2048→16384→2048, Expert 1: 1024→4096→1024)
    # 9. Residual

    pass  # Automatically handled by nn.scan

# ✅ After 18 layers:
# xs[0]: [4, 784, 2048]  (Prefix, fully processed)
# xs[1]: [4, 33, 1024]   (Suffix, fully processed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 8 요약:
  Layer 0의 구조를 17번 더 반복 (총 18 layers).
  - 각 layer마다 고유한 가중치를 가짐 (nn.scan으로 효율적 구현)
  - 매 layer마다 prefix↔suffix 간 cross-attention이 일어남
    → 깊어질수록 이미지/언어 정보가 action 토큰에 점점 더 녹아듦
  - 18층을 거치면서 action 토큰은 "현재 관찰에 맞는 행동 속도"를 표현
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 9: Final Layer Normalization

**코드 위치**: `src/openpi/models/gemma.py:409-411`

```python
# ═══════════════════════════════════════════════════════════════
# Step 9: Final RMSNorm (Expert별)
# ═══════════════════════════════════════════════════════════════

outputs = []
for i, (x, final_norm) in enumerate(zip(xs, self.final_norms)):
    if x is not None:
        x_final, _ = final_norm(x, None)  # adarms_cond=None (π₀)
        outputs.append(x_final)

# ✅ Output:
# outputs[0]: [4, 784, 2048]  (Prefix final output)
# outputs[1]: [4, 33, 1024]   (Suffix final output)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 9 요약:
  18개 Transformer layer를 모두 통과한 후 마지막 정규화.
  - 각 expert마다 독립적인 final RMSNorm 가중치 적용
  - 이후 action 예측에만 Suffix(Expert 1) 출력이 사용됨
  - Prefix(Expert 0) 출력은 학습 시에는 사용되지 않음
    (추론 시에도 KV Cache에 이미 반영되어 있어 별도 처리 불필요)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 10: Velocity Prediction

**코드 위치**: `src/openpi/models/pi0.py:212`

```python
# ═══════════════════════════════════════════════════════════════
# Step 10: Action Tokens → Velocity Prediction
# ═══════════════════════════════════════════════════════════════

# suffix_out: [4, 33, 1024]
# 마지막 action_horizon(32)개 토큰만 추출 (state 토큰 제외)
action_output = suffix_out[:, -32:]  # [4, 32, 1024]

# Project to action dimension
v_t = self.action_out_proj(action_output)
# Linear(1024 → 7)
# v_t: [4, 32, 7]

# ✅ Output:
# v_t: [4, 32, 7]  (Predicted velocity field)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 10 요약:
  Transformer를 통과한 action 토큰을 실제 행동 차원으로 변환.
  - suffix_out[:, -32:]: 33개 중 마지막 32개만 추출 (state 토큰 제외)
  - Linear(1024→7): Action Expert 차원 → 로봇 DoF 차원
  - 출력 v_t는 Flow Matching에서의 "속도(velocity)"
    = 현재 x_t에서 어느 방향으로 얼마나 이동해야 하는지
  - 학습: 이 v_t와 정답 u_t의 차이로 loss 계산
  - 추론: 이 v_t를 Euler step에 사용하여 x_t를 업데이트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# 직선 경로이므로 t에 무관한 상수

# ─── Loss ───
loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
# MSE loss averaged over action dimensions
# loss: [4, 32]  (timestep별 loss)

# ✅ Output:
# loss: [4, 32] (training objective)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 11 요약:
  Flow Matching 학습 목표: 모델이 예측한 속도와 정답 속도의 MSE.
  - 정답 속도 u_t = noise - actions  (직선 보간 경로의 접선 벡터)
    t와 무관한 상수 → 어떤 t에서 샘플링해도 동일한 방향
  - 손실 = ||v_t - u_t||^2  (L2 거리)
  - 이 loss를 역전파하면 모델은 "noise → data 방향"을 학습
  - 학습 전 과정이 단 1번의 forward pass로 끝남
    (이유: ground truth actions로 직접 x_t를 만들 수 있으므로)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Inference: Flow Matching Sampling

**코드 위치**: `src/openpi/models/pi0.py:217-279`

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
    # kv_cache: 18 layers × [4, 784, 1, 256]  ← 저장!

    # ─── Step I-2: 초기화 ───
    noise = jax.random.normal(rng, (4, 32, 7))
    x_t = noise  # time=1.0에서 시작 (pure noise)
    dt = -1.0 / num_steps  # -0.1

    # ─── Step I-3: Iterative Denoising ───
    def step(carry):
        x_t, time = carry
        # time: 1.0 → 0.9 → 0.8 → ... → 0.1 → 0.0

        # Suffix embedding (매 step x_t, time이 바뀌므로 재계산)
        suffix_tokens, suffix_mask, suffix_ar_mask, _ = \
            self.embed_suffix(observation, x_t, jnp.broadcast_to(time, [4]))
        # suffix_tokens: [4, 33, 1024]

        # Attention mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=33)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        # [4, 33, 817]

        # Positions
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Transformer (Expert 1만, KV cache 재사용!)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # Prefix는 None (cache에서 가져옴)
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,  # ← 저장된 cache 재사용!
            adarms_cond=[None, None],
        )

        # Velocity 예측
        v_t = self.action_out_proj(suffix_out[:, -32:])
        # Linear(1024 → 7): [4, 32, 7]

        # Euler integration: x_{t+dt} = x_t + dt * v_t
        return x_t + dt * v_t, time + dt

    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2  # time > 0

    # While loop (10 iterations)
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

    return x_0  # [4, 32, 7]  ← Denoised actions!

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Inference 요약:
  Pure noise에서 시작하여 Euler integration으로 clean action을 복원.
  [Phase 1] Prefix KV Cache (1회 실행):
    - 이미지/텍스트 토큰을 한 번만 Transformer에 통과
    - 18 layers × [4, 784, 1, 256] KV 값을 메모리에 저장
    - 추론 내내 관찰(이미지/언어)은 변하지 않으므로 재계산 불필요

  [Phase 2] Denoising Loop (10회 반복):
    for t in [1.0, 0.9, ..., 0.1]:
      1. 현재 x_t와 t로 suffix 임베딩 재생성  ← x_t, t가 매번 바뀜
      2. KV Cache + suffix 토큰으로 Transformer 실행 (Expert 1만)
      3. v_t = 예측된 속도  (현재 위치에서 data 방향)
      4. x_{t+dt} = x_t + (-0.1) × v_t  (Euler step)
    x_0 = 최종 action  (noise → clean action)

  핵심: prefix 1회 + suffix 10회 = 총 11회 Transformer 실행
        (매번 전체 재계산하면 110회 → KV Cache로 10배 절약)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
│ Images:  3 × [4, 256, 2048]    (SigLIP → 2048 proj)       │
│ Text:    [4, 16, 2048]         (Embedder)                  │
│ ─────────────────────────────────────────                  │
│ Prefix:  [4, 784, 2048]        (Concatenated)              │
│          Expert 0 (PaliGemma 2B) 처리                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               Step 5: Suffix Embedding                      │
├─────────────────────────────────────────────────────────────┤
│ State:   [4, 7]  → Linear(7→1024)        → [4, 1, 1024]   │
│ Actions: [4,32,7]→ Linear(7→1024)        → [4, 32, 1024]  │
│ Time:    [4]     → sincos PE + MLP concat → 액션에 혼합     │
│ ─────────────────────────────────────────                  │
│ Suffix:  [4, 33, 1024]    (state 1 + action 32)            │
│          Expert 1 (Action Expert 300M) 처리                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 6: Attention Mask Generation                  │
├─────────────────────────────────────────────────────────────┤
│ Mask:      [4, 817, 817]                                   │
│ Positions: [4, 817]                                        │
│                                                             │
│ 그룹:  prefix(0) ← state(1) ← action(2)                   │
│        prefix↔prefix 양방향                                 │
│        action↔action 양방향                                 │
│        action→prefix/state 가능, prefix→suffix 차단        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│        Step 7-8: Multi-Expert Transformer (18 layers)       │
├─────────────────────────────────────────────────────────────┤
│ xs[0]: [4, 784, 2048] ──→ ... ──→ [4, 784, 2048]          │
│        Expert 0 (PaliGemma): QKV 2048→256, FFN 2048→16384  │
│                                                             │
│ xs[1]: [4, 33, 1024]  ──→ ... ──→ [4, 33, 1024]           │
│        Expert 1 (Action):   QKV 1024→256, FFN 1024→4096    │
│                                                             │
│ Attention: Q,K concat → [4, 817, 8, 256] (공유 계산)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│             Step 9: Final Normalization                     │
├─────────────────────────────────────────────────────────────┤
│ Prefix:  [4, 784, 2048]                                    │
│ Suffix:  [4, 33, 1024]                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 10: Velocity Prediction                       │
├─────────────────────────────────────────────────────────────┤
│ suffix_out[:, -32:] → Linear(1024→7) → v_t: [4, 32, 7]   │
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
# Expert 0 (PaliGemma, width=2048):
# - "q_einsum":          [8, 2048, 256]
# - "kv_einsum":         [2, 1, 2048, 256]
# - "attn_vec_einsum":   [8, 256, 2048]
# - "mlp/gating_einsum": [2, 2048, 16384]
# - "mlp/linear":        [16384, 2048]

# Expert 1 (Action Expert, width=1024):
# - "q_einsum_1":          [8, 1024, 256]     ← 다른 width!
# - "kv_einsum_1":         [2, 1, 1024, 256]
# - "attn_vec_einsum_1":   [8, 256, 1024]
# - "mlp/gating_einsum_1": [2, 1024, 4096]
# - "mlp/linear_1":        [4096, 1024]

# 공통: head_dim=256, num_heads=8, num_kv_heads=1, depth=18
# → 서로 다른 차원의 입력을 같은 256dim으로 projection 후 attention 공유
```

### 2. π₀ Suffix 구성

```python
# state_proj: Linear(7 → 1024) → [4, 1, 1024]  (1개 state 토큰)
# action_in_proj: Linear(7 → 1024) → [4, 32, 1024]
# time_emb: sincos PE → [4, 1024]
#
# action + time 결합 (MLP):
# concat([action_tokens, time_tokens]) → [4, 32, 2048]
# → action_time_mlp_in:  Linear(2048→1024) + SiLU
# → action_time_mlp_out: Linear(1024→1024)
# → [4, 32, 1024]
#
# suffix = concat([state_token, action_tokens]) → [4, 33, 1024]
```

### 3. Flow Matching

```python
# Training:
time ~ Beta(1.5, 1.0)  # [0, 1], t=1 is noise, t=0 is data
x_t = time * noise + (1 - time) * actions
u_t = noise - actions  # Target velocity (상수)
loss = ||v_t - u_t||^2

# Inference (Euler integration):
x_t = noise  # t=1
for t in [1.0, 0.9, 0.8, ..., 0.1]:
    v_t = model(x_t, t)
    x_t = x_t + (-0.1) * v_t  # Euler step
# x_0 = clean actions
```

### 4. Attention Pattern

```
prefix(cumsum=0):  prefix끼리 양방향, suffix를 볼 수 없음
state(cumsum=1):   prefix 참조 가능, action을 볼 수 없음
action(cumsum=2):  prefix + state + action 전부 참조 가능 (양방향)
```

### 5. KV Cache Reuse (Inference)

```python
# Prefix를 한 번만 처리:
_, kv_cache = llm([prefix_tokens, None], ...)
# 18 layers × [4, 784, 1, 256] 저장

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

- 2026-02-14: π₀ 전용으로 수정
  - π₀.₅ AdaRMS 관련 내용 제거
  - Action Expert width 2048 → 1024 수정
  - State 토큰 추가 (suffix 32 → 33 토큰)
  - Action+Time MLP 방식으로 수정
  - Attention mask 차원 816 → 817 수정
  - 시각화 및 설명 전반 수정

---

**작성자**: AI Analysis
**프로젝트**: openpi (Physical Intelligence)
**버전**: 2.0
