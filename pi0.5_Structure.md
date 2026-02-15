# π₀.₅ 모델 데이터 흐름 Step-by-Step 완전 가이드

이 문서는 π₀.₅ 모델에서 **입력 데이터가 어떻게 처리되어 최종 출력이 되는지**를 한 단계씩 추적합니다.
  
> **📌 π₀ vs π₀.₅ 핵심 차이점**
>
> | | π₀ | π₀.₅ |
> |---|---|---|
> | **State 입력 방식** | continuous suffix 토큰 (1개) | 이산 언어 토큰으로 텍스트와 함께 prefix에 포함 |
> | **Timestep 주입** | Action + Time concat → MLP (suffix 내부) | **AdaRMSNorm** — Action Expert의 모든 layer에 조건 주입 |
> | **Suffix 구성** | `[state(1), action(32)]` = 33 tokens | `[action(50)]` = 50 tokens (state 없음) |
> | **max_token_len** | 48 | 200 (state가 텍스트로 들어가므로 더 길어짐) |
> | **action_horizon** | 50 (기본값 동일) | 50 |
> | **action_dim** | 32 (기본값 동일) | 32 |

> **📌 학습 vs 추론 구분**
> - 🏋️ **학습**: Ground truth actions + noise → Flow Matching loss 계산
> - 🎯 **추론**: Pure noise에서 시작 → 10회 Euler integration으로 action 생성

Step 0: 원본 입력 데이터 (Images, State-as-text, Actions)  
Step 1: Observation 객체 생성 (State → 토큰화, uint8 → float32 정규화)  
Step 2: Image Embedding (SigLIP) - 3×256 = 768 tokens    
Step 3: State+Text Embedding (Gemma Embedder) - 최대 200 tokens  
Step 4: Prefix Concatenation - 최대 968 tokens (Image + State/Text)  
Step 5: Action Embedding (Suffix) - 50 tokens + AdaRMSNorm 조건 생성  
Step 6: Attention Mask 생성 - [4, 1018, 1018]  
Step 7: Transformer Layer 0 상세 분석 (AdaRMSNorm 포함)  
7-1: Pre-Attention AdaRMSNorm (Adaptive, time_emb 조건부)  
7-2: QKV Projection (Multi-Expert)  
7-3: RoPE (Rotary Position Embedding)  
7-4: Grouped Query Attention  
7-5: Output Projection (Expert별)  
7-6: Gated Residual Connection (AdaRMSNorm gate)  
7-7: FeedForward Network (AdaRMSNorm gate 적용)    
Step 8: Transformer Layers 1-17 (18 layers total)  
Step 9: Final Layer Normalization (AdaRMSNorm)  
Step 10-11: Velocity Prediction + Flow Matching Loss  


**예시 데이터**:
- Batch Size: B = 4  
- Images: 3개 (base_0, left_wrist_0, right_wrist_0)  
- Text + State: 최대 200 tokens (state가 텍스트 토큰으로 인코딩됨)  
- Actions: 50 timesteps, 32 DoF  
- Model: π₀.₅ (`pi05=True`)  

---

## 📍 Step 0: 원본 입력 데이터

```python
# ═══════════════════════════════════════════════════════════════
# Step 0: Raw Input (Python Dictionary) — π₀.₅
# ═══════════════════════════════════════════════════════════════

raw_input = {
    # ─── Images ───
    "image": {
        "base_0_rgb":        np.array([4, 224, 224, 3], dtype=uint8),
        "left_wrist_0_rgb":  np.array([4, 224, 224, 3], dtype=uint8),
        "right_wrist_0_rgb": np.array([4, 224, 224, 3], dtype=uint8),
    },
    "image_mask": {
        "base_0_rgb":        np.array([True, True, True, True]),
        "left_wrist_0_rgb":  np.array([True, True, True, True]),
        "right_wrist_0_rgb": np.array([True, True, True, True]),
    },

    # ─── Robot State (π₀.₅에서는 텍스트 토큰으로 변환됨) ───
    # 예: "state: [0.12, -0.34, ..., 0.56]" 형태의 문자열이
    # 언어 명령 prompt와 함께 tokenized_prompt 에 합산됨
    # 아래는 이미 policy_config.py 의 전처리 후 상태

    # ─── Tokenized Prompt (State + Language, 합쳐서 텍스트화) ───
    "tokenized_prompt": np.array([
        [15234, 67, 123, ..., 8821, 0, 0, ...],  # "state: [0.12, ...] pick up fork" + padding
        ...
    ], dtype=int32),  # [4, 200]
    "tokenized_prompt_mask": np.array([
        [True, True, ..., True, False, ...],  # 실제 토큰은 True, padding은 False
        ...
    ], dtype=bool),  # [4, 200]

    # ─── Actions (Training only) ───
    "actions": np.array([4, 50, 32], dtype=float32),  # Ground truth actions
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 0 요약 (π₀.₅ 핵심 차이):
  π₀.₅에서 가장 중요한 변화: State가 더 이상 별도 suffix 토큰이 아님.
  - State는 텍스트로 직렬화되어 language prompt 뒤에 붙어
    하나의 tokenized_prompt로 합쳐진다.
  - 예: "pick up the cup\nstate: 0.12, -0.34, 0.56, ..."
  - 이로 인해 max_token_len이 48 → 200으로 증가
  - Actions: 50 timesteps, 32 DoF (π₀보다 horizon이 길고 DoF가 많음)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **🔄 학습 vs 추론**
> | | 학습 (Training) | 추론 (Inference) |
> |---|---|---|
> | **Images** | 동일 | 동일 |
> | **tokenized_prompt** | 동일 | 동일 |
> | **Actions** | ✅ Ground truth 필요 | ❌ 없음 (noise에서 생성) |

---

## 📍 Step 1: Observation 객체 생성

**코드 위치**: `src/openpi/models/model.py` + `src/openpi/policies/policy_config.py`

```python
# ═══════════════════════════════════════════════════════════════
# Step 1: Dictionary → Observation Object (π₀.₅)
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
        "base_0_rgb":        [4, 224, 224, 3],  # float32, [-1, 1]
        "left_wrist_0_rgb":  [4, 224, 224, 3],
        "right_wrist_0_rgb": [4, 224, 224, 3],
    },
    image_masks={
        "base_0_rgb":        [4],  # bool
        "left_wrist_0_rgb":  [4],
        "right_wrist_0_rgb": [4],
    },
    state=[4, 32],                 # float32 (내부 참조용, suffix에는 사용 안 함)
    tokenized_prompt=[4, 200],     # int32  ← π₀(16) 보다 훨씬 김
    tokenized_prompt_mask=[4, 200] # bool
)

# ✅ Output Shape:
# - Images: 3개 × [4, 224, 224, 3] float32 [-1, 1]
# - State:  [4, 32] float32  (π₀.₅에서는 prefix에 포함되어 있으므로 suffix에 별도 주입 없음)
# - tokenized_prompt: [4, 200] int32

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 1 요약 (π₀ vs π₀.₅):
  π₀:   state = [4, 7]  → suffix에서 Linear(7→1024)로 1개 state 토큰 생성
  π₀.₅: state = [4, 32] → 텍스트로 직렬화되어 tokenized_prompt[4, 200]에 이미 포함
        따라서 suffix 생성 시 state 토큰을 별도로 추가하지 않음
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 2: Image Embedding (SigLIP)

**코드 위치**: `src/openpi/models/pi0.py:113-125` + `src/openpi/models/siglip.py`

```python
# ═══════════════════════════════════════════════════════════════
# Step 2: Images → Image Tokens (SigLIP Vision Encoder)
# π₀.₅는 π₀와 완전히 동일한 SigLIP 구조 사용
# ═══════════════════════════════════════════════════════════════

image_tokens_list = []
for image_name in observation.images:
    image = observation.images[image_name]  # [4, 224, 224, 3]

    image_tokens, _ = self.PaliGemma.img(image, train=False)

    # ─── SigLIP 내부 처리 (π₀와 동일) ───
    # 2-1. Patch Embedding: [4, 224, 224, 3] → patches [4, 256, 588]
    # 2-2. Positional Embedding (Sinusoidal 2D)
    # 2-3. Transformer Encoder (27 layers, So400m/14)
    #       - width: 1152, heads: 16, head_dim: 72
    # 2-4. Final Projection: nn.Dense(1152 → 2048)

    image_tokens_list.append(image_tokens)  # [4, 256, 2048]

# ✅ Output:
# image_tokens_list = [
#     [4, 256, 2048],  # base_0_rgb
#     [4, 256, 2048],  # left_wrist_0_rgb
#     [4, 256, 2048],  # right_wrist_0_rgb
# ]
# Total: 3 × 256 = 768 image tokens

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 2 요약:
  π₀.₅도 π₀와 완전히 동일한 SigLIP (ViT-So400m/14) 사용.
  - 224×224 이미지 → 14×14 크기 패치 256개 → 27층 ViT → 2048차원
  - 3개 카메라 각각 독립 처리 → 총 768개 이미지 토큰
  - 차이 없음
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 3: Text/State Embedding (Gemma Embedder)

**코드 위치**: `src/openpi/models/gemma.py:148-154` + `pi0.py:128-133`

```python
# ═══════════════════════════════════════════════════════════════
# Step 3: Token IDs → Text Embeddings (π₀.₅)
# π₀.₅는 tokenized_prompt 가 state 정보를 이미 포함
# ═══════════════════════════════════════════════════════════════

# 입력: observation.tokenized_prompt
# [4, 200] int32  ← π₀는 [4, 16] 이었음
tokenized_prompt = observation.tokenized_prompt  # [4, 200]

tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")

# ─── Embedder 내부 ───
# 1. Embedding table lookup: [257152, 2048]
# x = embedding_table[(tokenized_prompt,)]  # [4, 200, 2048]
# 2. Scale: x *= √2048 ≈ 45.25

# ✅ Output:
# text_tokens: [4, 200, 2048]
#
# 내용 예시 (π₀.₅):
# tokens[0:5]   → "pick up the fork" 언어 명령
# tokens[5:...]  → "state: 0.12, -0.34, 0.56, ..." 상태 텍스트
# tokens[-1~]   → padding

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 3 요약 (π₀ vs π₀.₅):
  π₀:   text_tokens: [4, 16, 2048]   (언어 명령만)
  π₀.₅: text_tokens: [4, 200, 2048]  (언어 명령 + state 텍스트 직렬화)

  State를 텍스트로 표현하면:
  - 장점: 연속값을 언어 모델의 강력한 표현력으로 처리 가능
  - 장점: state 차원이 달라도 범용적으로 처리 가능 (다양한 로봇 지원)
  - 단점: 토큰 수가 늘어남 (16 → 200)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 4: Prefix Token Concatenation

**코드 위치**: `src/openpi/models/pi0.py:106-137`

```python
# ═══════════════════════════════════════════════════════════════
# Step 4: Image + Text/State → Prefix Sequence (π₀.₅)
# ═══════════════════════════════════════════════════════════════

def embed_prefix(self, obs):
    tokens = []
    input_mask = []
    ar_mask = []

    # ─── 4-1: Image tokens 추가 (π₀와 동일) ───
    for name in obs.images:
        image_tokens = image_tokens_list.pop(0)  # [4, 256, 2048]
        tokens.append(image_tokens)
        mask = einops.repeat(obs.image_masks[name], "b -> b s", s=256)  # [4, 256]
        input_mask.append(mask)
        ar_mask += [False] * 256  # 양방향 attention

    # ─── 4-2: Text+State tokens 추가 ───
    if obs.tokenized_prompt is not None:
        text_tokens = tokenized_inputs  # [4, 200, 2048]  ← π₀는 [4, 16, 2048]
        tokens.append(text_tokens)
        input_mask.append(obs.tokenized_prompt_mask)  # [4, 200]
        ar_mask += [False] * 200  # 양방향 attention

    # ─── 4-3: Concatenation ───
    prefix_tokens = jnp.concatenate(tokens, axis=1)
    # [4, 768, 2048] + [4, 200, 2048] = [4, 968, 2048]
    #  ^^^^^^^^^^^^     ^^^^^^^^^^^^
    #  Image (3×256)    Text + State (200)

    prefix_mask = jnp.concatenate(input_mask, axis=1)  # [4, 968]
    prefix_ar_mask = jnp.array(ar_mask)                # [968] (전부 False)

    return prefix_tokens, prefix_mask, prefix_ar_mask

# ✅ Output:
# - prefix_tokens: [4, 968, 2048]
#   ├─ Image 0 (base):          tokens[0:256]     (256개)
#   ├─ Image 1 (left_wrist):    tokens[256:512]   (256개)
#   ├─ Image 2 (right_wrist):   tokens[512:768]   (256개)
#   └─ Text + State:            tokens[768:968]   (200개)
# - prefix_mask: [4, 968]
# - prefix_ar_mask: [968] (all False = bidirectional)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 4 요약 (π₀ vs π₀.₅):
  π₀:   prefix = [image(768), text(16)]   = 784 tokens
  π₀.₅: prefix = [image(768), text+state(200)] = 968 tokens

  State가 prefix에 합류함으로써:
  - Suffix는 이제 오직 action 토큰만으로 구성됨
  - Prefix↔Suffix attention을 통해 action이 state 정보를 참조 가능
  - Prefix 내부는 모두 양방향 attention (ar_mask 전부 False)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 5: Action Embedding + AdaRMSNorm 조건 생성 (Suffix)

**코드 위치**: `src/openpi/models/pi0.py:139-186`

```python
# ═══════════════════════════════════════════════════════════════
# Step 5: Actions → Suffix Tokens + time_emb (π₀.₅ 방식)
# π₀.₅의 핵심: State 토큰 없음, Timestep은 AdaRMSNorm으로 주입
# ═══════════════════════════════════════════════════════════════

# Training input:
actions = raw_input["actions"]  # [4, 50, 32]
state   = observation.state     # [4, 32]  (suffix에 사용 안 함)

# ─── 5-1: Flow Matching Preparation (π₀와 동일) ───
noise = jax.random.normal(noise_rng, actions.shape)  # [4, 50, 32]
time = jax.random.beta(time_rng, 1.5, 1.0, batch_shape=[4]) * 0.999 + 0.001
# time: [4]  예: [0.234, 0.891, 0.456, 0.123]

time_expanded = time[:, None, None]  # [4, 1, 1]
x_t = time_expanded * noise + (1 - time_expanded) * actions  # [4, 50, 32]
u_t = noise - actions  # [4, 50, 32]  (target velocity)

# ─── 5-2: Action Token Projection ───
action_tokens = self.action_in_proj(x_t)
# Linear(32 → 1024)
# [4, 50, 32] → [4, 50, 1024]

# ─── 5-3: Timestep Embedding (Sinusoidal) ───
def posemb_sincos(pos, embedding_dim=1024, min_period=4e-3, max_period=4.0):
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # [512]
    period = min_period * (max_period / min_period) ** fraction

    sinusoid_input = jnp.einsum("i,j->ij", pos, 1.0 / period * 2 * jnp.pi)  # [4, 512]
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
    # [4, 1024]

time_emb = posemb_sincos(time, 1024, min_period=4e-3, max_period=4.0)
# time_emb: [4, 1024]  ← 스칼라 t → 1024차원

# ─── 5-4: Time MLP (π₀.₅ 전용) ── AdaRMSNorm 입력 생성 ───
# π₀:   time_emb를 action과 concat 후 MLP로 action에 직접 혼합
# π₀.₅: time_emb를 별도 MLP로 처리 → adarms_cond 로 각 layer에 주입
time_emb = self.time_mlp_in(time_emb)    # Linear(1024 → 1024): [4, 1024]
time_emb = nnx.swish(time_emb)
time_emb = self.time_mlp_out(time_emb)   # Linear(1024 → 1024): [4, 1024]
time_emb = nnx.swish(time_emb)
adarms_cond = time_emb                   # [4, 1024] ← 각 Transformer layer의 AdaRMS 조건

# ─── 5-5: Suffix 구성 (π₀.₅: state 토큰 없음) ───
# π₀:   suffix = [state_token(1), action_tokens(32)] = 33 tokens
# π₀.₅: suffix = [action_tokens(50)]                 = 50 tokens
suffix_tokens = action_tokens  # [4, 50, 1024]
suffix_mask   = jnp.ones([4, 50], dtype=bool)

# AR Mask:
# - 첫 번째 action 토큰: True  (prefix가 이 토큰을 볼 수 없음)
# - 나머지 action 토큰: False  (action끼리 양방향 attention)
suffix_ar_mask = jnp.array([True] + [False] * 49)
# [True, False, False, ..., False]
#  act0  act1   act2        act49

# ✅ Output:
# - suffix_tokens: [4, 50, 1024]    ← π₀는 [4, 33, 1024]
# - suffix_mask:   [4, 50]
# - suffix_ar_mask:[50] ([True, False×49])
# - adarms_cond:   [4, 1024]        ← π₀는 None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 5 요약 (π₀ vs π₀.₅):

  π₀ Suffix 구성:
    state_proj:            Linear(7→1024) → 1개 state 토큰
    action + time concat:  [action(1024), time(1024)] → MLP → action 토큰에 time 혼합
    suffix:                [state(1), action(32)] = 33 tokens
    adarms_cond:           None

  π₀.₅ Suffix 구성:
    action_in_proj:        Linear(32→1024) → 50개 action 토큰 (time 혼합 없음)
    time_mlp:              sincos(t) → Linear → SiLU → Linear → SiLU
                           → adarms_cond [4, 1024]  (각 layer에 조건으로 전달)
    suffix:                [action(50)] = 50 tokens (state 없음)

  핵심 설계 철학의 차이:
    π₀:   "time을 action에 직접 섞는다"  (suffix level에서 처리)
    π₀.₅: "time을 각 layer의 normalization에 조건으로 건다"  (layer level에서 처리)
          → AdaRMSNorm이 매 layer마다 t에 따라 scale/shift/gate를 동적 조정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **🔄 학습 vs 추론**
> | | 학습 (Training) | 추론 (Inference) |
> |---|---|---|
> | **입력 actions** | Ground truth actions | 없음 (noise에서 시작) |
> | **Noise** | `noise ~ N(0, I)` | `noise ~ N(0, I)` (= 초기 x_t) |
> | **Timestep t** | `t ~ Beta(1.5, 1.0)*0.999+0.001` | `t = 1.0, 0.9, ..., 0.0` |
> | **adarms_cond** | time_emb [4, 1024] | 매 step 재계산 (t 바뀜) |

---

## 📍 Step 6: Attention Mask 생성

**코드 위치**: `src/openpi/models/pi0.py:19-44` + `202-208`

```python
# ═══════════════════════════════════════════════════════════════
# Step 6: Create Attention Mask (π₀.₅)
# ═══════════════════════════════════════════════════════════════

# ─── 6-1: Concatenate masks ───
input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
# [4, 968] + [4, 50] = [4, 1018]
#  ^^^^^^^^   ^^^^^^
#  Prefix      Suffix(action only)

ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
# [968] + [50] = [1018]
# ar_mask = [False×968, True, False×49]
#            ^^^^^^^^    ^^^^  ^^^^^^^^
#            Prefix      act0  act1~49

# ─── 6-2: cumsum으로 그룹 분리 ───
cumsum = jnp.cumsum(ar_mask)
# [0×968, 1, 1×49]
# → 그룹 0: prefix (이미지 + 텍스트/state)
# → 그룹 1: action (모두 동일, 양방향)

# π₀:   3개 그룹 (prefix=0, state=1, action=2)
# π₀.₅: 2개 그룹 (prefix=0, action=1) ← state가 prefix에 포함되어 사라짐

# ─── 6-3: Attention 패턴 시각화 ───
"""
Attention pattern [1018, 1018]:

                  Prefix(968)           Actions(50)
              cumsum=0                  cumsum=1
         ┌──────────────────────────┬──────────────┐
Prefix   │  ✓  ✓  ✓  ...  ✓        │  ✗  ...  ✗  │ cumsum=0
(0-967)  │  (양방향, img+text+state) │              │
         ├──────────────────────────┼──────────────┤
Actions  │  ✓  ✓  ✓  ...  ✓        │  ✓  ...  ✓  │ cumsum=1
(968-1017)│  prefix 전체 참조 가능   │  (양방향)    │
         └──────────────────────────┴──────────────┘

규칙: cumsum[key] <= cumsum[query] → 참조 가능
  prefix→prefix:   0<=0 ✓ 양방향
  prefix→action:   1<=0 ✗ 차단  (prefix는 suffix에 영향받지 않음)
  action→prefix:   0<=1 ✓ (이미지, 언어, state 모두 참조 가능)
  action→action:   1<=1 ✓ 양방향
"""

attn_mask = make_attn_mask(input_mask, ar_mask)
# attn_mask: [4, 1018, 1018]
positions = jnp.cumsum(input_mask, axis=1) - 1  # [4, 1018]

# ✅ Output:
# - attn_mask: [4, 1018, 1018] bool
# - positions: [4, 1018] int32

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 6 요약 (π₀ vs π₀.₅):
  π₀:   [817, 817] — prefix(0) / state(1) / action(2) 3그룹
  π₀.₅: [1018, 1018] — prefix(0) / action(1) 2그룹

  π₀.₅가 더 단순한 이유:
  - state가 prefix에 있으므로 별도 그룹 불필요
  - action은 prefix 전체(이미지+언어+state)를 한꺼번에 참조
  - prefix는 suffix를 볼 수 없음 (KV Cache 재사용 가능)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 7: Multi-Expert Transformer Layer 0 (AdaRMSNorm)

이제 18개의 Transformer layer 중 **첫 번째 layer**를 자세히 봅니다.
π₀.₅의 핵심 변화: Expert 1(Action Expert)에 **AdaRMSNorm** 적용.

**코드 위치**: `src/openpi/models/gemma.py:112-131`, `284-333`

### Step 7-1: Pre-Attention AdaRMSNorm

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-1: Pre-Attention Normalization
# π₀:   Expert 0, 1 모두 일반 RMSNorm (adarms_cond=None)
# π₀.₅: Expert 0 일반 RMSNorm, Expert 1 AdaRMSNorm (adarms_cond=time_emb)
# ═══════════════════════════════════════════════════════════════

# 입력:
xs = [prefix_tokens, suffix_tokens]
# xs[0]: [4, 968, 2048]  (Expert 0 - PaliGemma, width=2048)
# xs[1]: [4, 50, 1024]   (Expert 1 - Action Expert, width=1024)

adarms_cond = [None, time_emb]
# [None, [4, 1024]]
# Expert 0: 일반 RMSNorm (cond=None)
# Expert 1: AdaRMSNorm  (cond=time_emb)

pre_attn = []
gates = []

# ─── Expert 0: 일반 RMSNorm ───
var_0 = jnp.mean(jnp.square(xs[0].astype(float32)), axis=-1, keepdims=True)  # [4, 968, 1]
normed_0 = xs[0] * jnp.reciprocal(jnp.sqrt(var_0 + 1e-6))
scale_0 = param("scale", zeros_init(), (2048,))
x_norm_0 = normed_0 * (1 + scale_0)   # [4, 968, 2048]
gate_0 = None                          # π₀.₅ Expert 0는 gate 없음

# ─── Expert 1: AdaRMSNorm (π₀.₅ 핵심) ───
var_1 = jnp.mean(jnp.square(xs[1].astype(float32)), axis=-1, keepdims=True)  # [4, 50, 1]
normed_1 = xs[1] * jnp.reciprocal(jnp.sqrt(var_1 + 1e-6))

# AdaRMSNorm: time_emb → (scale, shift, gate)를 동적으로 생성
modulation = nn.Dense(1024 * 3, kernel_init=zeros_init)(time_emb)
# Dense(1024 → 3072): [4, 1024] → [4, 3072]
scale_1, shift_1, gate_1 = jnp.split(modulation[:, None, :], 3, axis=-1)
# scale_1, shift_1, gate_1: 각 [4, 1, 1024]

x_norm_1 = normed_1 * (1 + scale_1) + shift_1
# [4, 50, 1024] * [4, 1, 1024] + [4, 1, 1024] = [4, 50, 1024]
# scale_1, shift_1 은 모든 50개 action 토큰에 동일하게 적용

# ✅ Output:
# x_norm_0: [4, 968, 2048],  gate_0: None
# x_norm_1: [4, 50, 1024],   gate_1: [4, 1, 1024]  ← gate는 residual에서 사용

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-1 요약 (π₀ vs π₀.₅):
  π₀:   두 Expert 모두 일반 RMSNorm
          scale 파라미터만으로 정규화 조정
  π₀.₅: Expert 0 일반 RMSNorm, Expert 1 AdaRMSNorm
          AdaRMSNorm이 time_emb로부터 scale, shift, gate를 생성:
          - scale_1: 각 차원별 크기 조정 (t에 따라 다름)
          - shift_1: 각 차원별 편향 추가 (t에 따라 다름)
          - gate_1:  Residual 연결의 가중치 (다음 Step에서 사용)
          → 매 layer마다 t에 맞는 feature transformation이 가능
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-2: QKV Projection (Multi-Expert)

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-2: QKV Projection (π₀.₅ — π₀와 구조는 동일)
# ═══════════════════════════════════════════════════════════════

# ─── Expert 0 (PaliGemma, width=2048) ───
q_0 = q_einsum_0("BTD,NDH->BTNH", x_norm_0)   # [4, 968, 8, 256]
k_0, v_0 = kv_einsum_0("BSD,2KDH->2BSKH", x_norm_0)  # [4, 968, 1, 256]

# ─── Expert 1 (Action Expert, width=1024) ───
q_1 = q_einsum_1("BTD,NDH->BTNH", x_norm_1)   # [4, 50, 8, 256]
k_1, v_1 = kv_einsum_1("BSD,2KDH->2BSKH", x_norm_1)  # [4, 50, 1, 256]

# ✅ Q, K, V Output:
# Expert 0: q[4,968,8,256], k[4,968,1,256], v[4,968,1,256]  ← 2048→256
# Expert 1: q[4,50,8,256],  k[4,50,1,256],  v[4,50,1,256]   ← 1024→256

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-2 요약:
  π₀.₅도 π₀와 동일한 구조 (입력 차원만 다름):
  - Expert 0: 2048 → 256 (prefix 토큰: 968개)
  - Expert 1: 1024 → 256 (suffix 토큰: 50개, π₀는 33개)
  - head_dim=256, num_heads=8, num_kv_heads=1 (GQA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-3: RoPE

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-3: Concat + RoPE
# ═══════════════════════════════════════════════════════════════

q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs))
# q: [4, 1018, 8, 256]  (968 + 50)  ← π₀는 [4, 817, 8, 256]
# k: [4, 1018, 1, 256]
# v: [4, 1018, 1, 256]

q = _apply_rope(q, positions=positions)  # [4, 1018, 8, 256]
k = _apply_rope(k, positions=positions)  # [4, 1018, 1, 256]
q *= 256 ** -0.5  # 1/√head_dim 스케일링
```

### Step 7-4: Grouped Query Attention

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-4: GQA — 1018개 전체 토큰에 대해 계산
# ═══════════════════════════════════════════════════════════════

q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=1)
# [4, 1018, 8, 256] → [4, 1018, 1, 8, 256]

logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
# logits: [4, 1, 8, 1018, 1018]

masked_logits = jnp.where(attn_mask[:, None, None, :, :], logits, -2.38e38)
probs = jax.nn.softmax(masked_logits, axis=-1)  # [4, 1, 8, 1018, 1018]

encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
# encoded: [4, 1018, 8, 256]

# ✅ 핵심: action 토큰이 prefix의 모든 정보 (이미지 + 언어 + state)를 참조
```

### Step 7-5: Output Projection

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-5: Output Projection (Expert별 독립)
# ═══════════════════════════════════════════════════════════════

# Expert 0 (Prefix):
expert_encoded_0 = encoded[:, :968]   # [4, 968, 8, 256]
expert_out_0 = out_einsum_0("BTNH,NHD->BTD", expert_encoded_0)
# [4, 968, 2048]

# Expert 1 (Suffix):
expert_encoded_1 = encoded[:, 968:]   # [4, 50, 8, 256]
expert_out_1 = out_einsum_1("BTNH,NHD->BTD", expert_encoded_1)
# [4, 50, 1024]

# ✅ Output:
# out[0]: [4, 968, 2048]
# out[1]: [4, 50, 1024]
```

### Step 7-6: Gated Residual Connection (π₀.₅ 핵심)

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-6: Gated Residual Connection
# π₀:   일반 덧셈  (gate=None)
# π₀.₅: Expert 1에 gate 적용  (gate=gate_1 from AdaRMSNorm)
# ═══════════════════════════════════════════════════════════════

def _gated_residual(x, y, gate):
    if gate is None:
        return x + y          # π₀, π₀.₅ Expert 0
    return x + y * gate       # π₀.₅ Expert 1 전용

# Expert 0 (일반 residual):
xs[0] = xs[0] + expert_out_0
# [4, 968, 2048] + [4, 968, 2048] = [4, 968, 2048]

# Expert 1 (gated residual):
xs[1] = xs[1] + expert_out_1 * gate_1
# [4, 50, 1024] + [4, 50, 1024] * [4, 1, 1024] = [4, 50, 1024]
# gate_1: [4, 1, 1024] → 모든 50개 action 토큰에 동일한 gate 적용

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-6 요약 (π₀ vs π₀.₅):
  π₀:   xs[i] = xs[i] + attn_out[i]                 (단순 합)
  π₀.₅: xs[1] = xs[1] + attn_out[1] * gate_1        (gate 가중 합)
        gate_1은 AdaRMSNorm이 time_emb에서 생성한 벡터
        → "현재 timestep t에서 attention 정보를 얼마나 반영할지" 동적 조정
        → t=1 (pure noise): gate가 큰 값 → attention 결과 많이 반영
        → t=0 (clean data): gate가 작은 값 → 기존 표현 유지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 7-7: FeedForward Network (AdaRMSNorm gate 적용)

```python
# ═══════════════════════════════════════════════════════════════
# Step 7-7: FFN with AdaRMSNorm gate (π₀.₅)
# ═══════════════════════════════════════════════════════════════

for i, (x, config) in enumerate(zip(xs, configs)):
    # ─── Pre-FFN AdaRMSNorm ───
    x_norm, gate = RMSNorm(name=f"pre_ffw_norm_{i}")(x, adarms_cond[i])
    # Expert 0: gate=None  (일반 RMSNorm)
    # Expert 1: gate=[4, 1, 1024]  (AdaRMSNorm, 새로운 gate 생성)

    # ─── GeGLU FeedForward ───
    if i == 0:  # Expert 0 (PaliGemma, width=2048, mlp_dim=16384)
        ff_gate = jnp.dot(x_norm, w_gating[0])    # [4, 968, 16384]
        ff1     = jnp.dot(x_norm, w_gating[1])    # [4, 968, 16384]
        activations = nn.gelu(ff_gate) * ff1        # [4, 968, 16384]
        outputs = jnp.dot(activations, w_linear)    # [4, 968, 2048]

    else:       # Expert 1 (Action Expert, width=1024, mlp_dim=4096)
        ff_gate = jnp.dot(x_norm, w_gating[0])    # [4, 50, 4096]
        ff1     = jnp.dot(x_norm, w_gating[1])    # [4, 50, 4096]
        activations = nn.gelu(ff_gate) * ff1        # [4, 50, 4096]
        outputs = jnp.dot(activations, w_linear)    # [4, 50, 1024]

# ─── Second Gated Residual ───
xs[0] = xs[0] + outputs[0]                          # 일반 residual
xs[1] = xs[1] + outputs[1] * gate_ffn               # gated residual

# ✅ Output (Layer 0 완료):
# xs[0]: [4, 968, 2048]
# xs[1]: [4, 50, 1024]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 7-7 요약:
  π₀.₅ Expert 1은 Pre-Attention과 Pre-FFN 두 번 AdaRMSNorm을 적용.
  각각 다른 (scale, shift, gate) 세트를 time_emb에서 생성.
  → 한 layer에서 AdaRMSNorm이 총 2회 적용 (attention 전, FFN 전)
  → 18 layers × 2 = 총 36회 timestep 조건 주입
  → π₀ (action+time MLP 1회) 보다 훨씬 세밀한 timestep 의존적 처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 8: Transformer Layers 1-17

**코드 위치**: `src/openpi/models/gemma.py:365-381` (nn.scan)

```python
# ═══════════════════════════════════════════════════════════════
# Step 8: Repeat Layer 0 for Layers 1-17
# ═══════════════════════════════════════════════════════════════

# nn.scan으로 18개 layer 반복 (각 layer 동일한 구조, 다른 weight)
for layer_idx in range(1, 18):
    # 매 layer마다 동일하게:
    # 1. Pre-Attention Norm
    #    - Expert 0: 일반 RMSNorm
    #    - Expert 1: AdaRMSNorm(time_emb) → (scale, shift, gate_attn)
    # 2. QKV (Expert 0: 2048→256, Expert 1: 1024→256)
    # 3. RoPE + concat → [4, 1018, 8, 256]
    # 4. GQA [4, 1, 8, 1018, 1018]
    # 5. Out Projection (Expert 0: 256→2048, Expert 1: 256→1024)
    # 6. Gated Residual
    #    - Expert 0: 단순 합
    #    - Expert 1: x + attn_out * gate_attn
    # 7. Pre-FFN Norm
    #    - Expert 0: 일반 RMSNorm
    #    - Expert 1: AdaRMSNorm(time_emb) → (scale, shift, gate_ffn)
    # 8. FFN (Expert 0: 2048→16384→2048, Expert 1: 1024→4096→1024)
    # 9. Gated Residual
    #    - Expert 0: 단순 합
    #    - Expert 1: x + ffn_out * gate_ffn
    pass

# ✅ After 18 layers:
# xs[0]: [4, 968, 2048]
# xs[1]: [4, 50, 1024]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 8 요약:
  π₀.₅는 18 layers × (attention AdaRMS + FFN AdaRMS) = 36회 time 조건 주입.
  매 layer를 거칠수록:
  - action 토큰이 image/언어/state prefix 정보를 깊이 통합
  - AdaRMSNorm gate가 현재 t에 맞게 feature 변환 강도를 조절
  - 결과적으로 t에 따른 세밀한 velocity 예측이 가능
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 9: Final Layer Normalization

**코드 위치**: `src/openpi/models/gemma.py:409-411`

```python
# ═══════════════════════════════════════════════════════════════
# Step 9: Final Normalization (π₀.₅)
# ═══════════════════════════════════════════════════════════════

outputs = []
for i, (x, final_norm) in enumerate(zip(xs, self.final_norms)):
    if x is not None:
        adarms_cond_i = adarms_cond[i]  # Expert 0: None, Expert 1: time_emb
        x_final, _ = final_norm(x, adarms_cond_i)
        # Expert 0: 일반 RMSNorm → x_final [4, 968, 2048]
        # Expert 1: AdaRMSNorm  → x_final [4, 50, 1024]
        outputs.append(x_final)

# ✅ Output:
# outputs[0]: [4, 968, 2048]
# outputs[1]: [4, 50, 1024]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 9 요약:
  π₀.₅도 Final Norm에서 AdaRMSNorm 적용 (총 37회 time 조건 주입).
  이후 action 예측에는 outputs[1] (Expert 1 Suffix) 만 사용.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 10: Velocity Prediction

**코드 위치**: `src/openpi/models/pi0.py:212`

```python
# ═══════════════════════════════════════════════════════════════
# Step 10: Action Tokens → Velocity Prediction (π₀.₅)
# ═══════════════════════════════════════════════════════════════

# suffix_out: [4, 50, 1024]
# π₀.₅는 state 토큰이 없으므로 전체 50개를 사용
action_output = suffix_out[:, -50:]  # [4, 50, 1024]
# (π₀는 suffix_out[:, -32:] 으로 state 토큰 제외)

v_t = self.action_out_proj(action_output)
# Linear(1024 → 32)  ← π₀는 Linear(1024 → 7)
# v_t: [4, 50, 32]

# ✅ Output:
# v_t: [4, 50, 32]  (Predicted velocity field)
#      ^  ^^  ^^
#      B  horizon  action_dim

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 10 요약 (π₀ vs π₀.₅):
  π₀:   suffix_out[:, -32:] → Linear(1024→7)  → v_t [4, 32, 7]
  π₀.₅: suffix_out[:, -50:] → Linear(1024→32) → v_t [4, 50, 32]
  → action_horizon 32→50, action_dim 7→32 으로 확장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Step 11: Flow Matching Loss

**코드 위치**: `src/openpi/models/pi0.py:214`

```python
# ═══════════════════════════════════════════════════════════════
# Step 11: Compute Flow Matching Loss (π₀.₅ — π₀와 동일한 방식)
# ═══════════════════════════════════════════════════════════════

u_t = noise - actions  # [4, 50, 32]  (target velocity, 직선 경로 접선)

loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
# MSE: [4, 50]  (batch × horizon 별 loss)

# ✅ Output:
# loss: [4, 50]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Step 11 요약:
  π₀와 완전히 동일한 Flow Matching 목표:
  loss = || v_t - (noise - actions) ||²
  차이는 shape뿐:
    π₀:   [4, 32, 7]
    π₀.₅: [4, 50, 32]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📍 Inference: Flow Matching Sampling

**코드 위치**: `src/openpi/models/pi0.py:217-279`

```python
# ═══════════════════════════════════════════════════════════════
# Inference: Iterative Denoising (π₀.₅)
# π₀와 구조는 동일하나 suffix/adarms_cond 방식이 다름
# ═══════════════════════════════════════════════════════════════

def sample_actions(self, rng, observation, num_steps=10):
    # ─── Phase 1: Prefix KV Cache (1회만 실행) ───
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    # prefix_tokens: [1, 968, 2048]  (배치=1)

    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )
    # kv_cache: 18 layers × [1, 968, 1, 256]  ← π₀보다 prefix가 더 길어 cache도 큼

    # ─── Phase 2: 초기화 ───
    noise = jax.random.normal(rng, (1, 50, 32))  # [1, 50, 32]  ← π₀는 [1, 32, 7]
    x_t = noise
    dt = -1.0 / num_steps  # -0.1

    # ─── Phase 3: Iterative Denoising (10회 반복) ───
    def step(carry):
        x_t, time = carry

        # Suffix embedding (π₀.₅ 방식)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, 1)
        )
        # suffix_tokens: [1, 50, 1024]   ← π₀는 [1, 33, 1024]
        # adarms_cond:   [1, 1024]        ← π₀는 None (time은 이미 suffix에 혼합됨)

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=50)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        # [1, 50, 968+50] = [1, 50, 1018]

        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Transformer (Expert 1만, KV cache 재사용!)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],  # ← π₀.₅만의 핵심: time 조건 전달
        )

        v_t = self.action_out_proj(suffix_out[:, -50:])
        # Linear(1024 → 32): [1, 50, 32]

        return x_t + dt * v_t, time + dt

    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0  # [1, 50, 32]  ← Denoised actions!

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Inference 요약 (π₀ vs π₀.₅):
  공통:
    [Phase 1] prefix KV Cache 1회 생성
    [Phase 2] 10회 Euler loop: suffix 재생성 → v_t 예측 → Euler step

  π₀.₅ 차이점:
    - prefix cache가 더 큼: 18 × [1, 968, 1, 256] (π₀는 18 × [1, 784, 1, 256])
    - suffix가 더 짧아짐:   [1, 50, 1024] (π₀는 [1, 33, 1024])
    - adarms_cond가 매 step 재계산: sincos(t) → time_mlp → [1, 1024]
      각 layer의 AdaRMSNorm에 전달 → t에 따른 세밀한 feature 조정
    - 출력: [1, 50, 32] (π₀는 [1, 32, 7])
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 전체 데이터 흐름 요약

### Shape 변화 추적

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Data (π₀.₅)                          │
├─────────────────────────────────────────────────────────────────┤
│ Images:            3 × [4, 224, 224, 3]  uint8 [0, 255]        │
│ State (as text):   [4, 32] → 직렬화 → tokenized_prompt에 포함   │
│ tokenized_prompt:  [4, 200]              int32                  │
│ Actions:           [4, 50, 32]           float32 (training)     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Step 1: Preprocessing                         │
├─────────────────────────────────────────────────────────────────┤
│ Images → float32 [-1, 1]                                        │
│ State 이미 tokenized_prompt에 포함됨                             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│             Step 2-4: Prefix Embedding (π₀.₅)                   │
├─────────────────────────────────────────────────────────────────┤
│ Images:  3 × [4, 256, 2048]    (SigLIP → 2048 proj)            │
│ Text+State: [4, 200, 2048]     (Embedder, state가 텍스트로 포함) │
│ ─────────────────────────────────────────────────────          │
│ Prefix:  [4, 968, 2048]        (Concatenated)                   │
│          Expert 0 (PaliGemma 2B) 처리                           │
│          π₀는 [4, 784, 2048]                                    │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│               Step 5: Suffix Embedding (π₀.₅)                   │
├─────────────────────────────────────────────────────────────────┤
│ Actions: [4,50,32] → Linear(32→1024) → [4, 50, 1024]          │
│ Time:    sincos(t)[1024] → time_mlp → adarms_cond[4, 1024]     │
│          (action에 직접 혼합 안 함 → AdaRMSNorm으로 전달)        │
│ ─────────────────────────────────────────────────────          │
│ Suffix:  [4, 50, 1024]   (action 50개, state 토큰 없음)         │
│          Expert 1 (Action Expert 300M) 처리                     │
│          π₀는 [4, 33, 1024] (state 1 + action 32)              │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│          Step 6: Attention Mask Generation (π₀.₅)               │
├─────────────────────────────────────────────────────────────────┤
│ Mask:      [4, 1018, 1018]                                      │
│ Positions: [4, 1018]                                            │
│                                                                  │
│ 2개 그룹: prefix(0) / action(1)                                  │
│          π₀는 3개 그룹: prefix(0) / state(1) / action(2)        │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│   Step 7-8: Multi-Expert Transformer (18 layers, π₀.₅)          │
├─────────────────────────────────────────────────────────────────┤
│ xs[0]: [4, 968, 2048] ──→ ... ──→ [4, 968, 2048]               │
│        Expert 0: 일반 RMSNorm, QKV 2048→256, FFN 2048→16384     │
│                                                                  │
│ xs[1]: [4, 50, 1024]  ──→ ... ──→ [4, 50, 1024]                │
│        Expert 1: AdaRMSNorm(time_emb), QKV 1024→256,            │
│                  FFN 1024→4096, Gated Residual                   │
│                                                                  │
│ AdaRMSNorm 조건 주입 횟수: 18 layers × 2 + final = 37회         │
│ Attention: Q,K concat → [4, 1018, 8, 256] (공유 계산)           │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│             Step 9: Final Normalization (π₀.₅)                  │
├─────────────────────────────────────────────────────────────────┤
│ Prefix:  [4, 968, 2048]  (일반 RMSNorm)                         │
│ Suffix:  [4, 50, 1024]   (AdaRMSNorm)                           │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│          Step 10: Velocity Prediction (π₀.₅)                    │
├─────────────────────────────────────────────────────────────────┤
│ suffix_out[:, -50:] → Linear(1024→32) → v_t: [4, 50, 32]      │
│ π₀는: suffix_out[:, -32:] → Linear(1024→7) → v_t: [4, 32, 7]  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│           Step 11: Loss Computation (π₀.₅)                      │
├─────────────────────────────────────────────────────────────────┤
│ loss = MSE(v_t, noise - actions)                                │
│      = mean((v_t - (noise - actions))^2, axis=-1)              │
│ loss: [4, 50]                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔥 π₀ vs π₀.₅ 전체 비교

### 아키텍처 차이

```python
# ──────────────── π₀ ────────────────
Pi0Config(
    pi05=False,
    paligemma_variant="gemma_2b",      # PaliGemma 2B (width=2048)
    action_expert_variant="gemma_300m", # Action Expert 300M (width=1024)
    action_dim=32,
    action_horizon=50,
    max_token_len=48,                  # 짧음 (텍스트만)
    discrete_state_input=False,        # state는 suffix에 continuous 토큰
)
# state 처리: Linear(action_dim→1024) → 1개 suffix 토큰
# time 처리:  sincos(t) + action concat → MLP → action 토큰에 혼합

# ──────────────── π₀.₅ ────────────────
Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b",      # 동일
    action_expert_variant="gemma_300m", # 동일
    action_dim=32,
    action_horizon=50,
    max_token_len=200,                 # 길어짐 (텍스트+state)
    discrete_state_input=True,         # state는 prefix에 텍스트 토큰
)
# state 처리: 텍스트 직렬화 → tokenized_prompt 에 포함 → prefix
# time 처리:  sincos(t) → time_mlp → adarms_cond → 각 layer AdaRMSNorm
```

### 시퀀스 길이 비교

```
π₀:
  Prefix:  image(768) + text(16)          = 784 tokens
  Suffix:  state(1)   + action(32)        =  33 tokens
  Total:                                  = 817 tokens
  Attention mask: [817, 817]

π₀.₅:
  Prefix:  image(768) + text+state(200)   = 968 tokens
  Suffix:  action(50)                     =  50 tokens
  Total:                                  = 1018 tokens
  Attention mask: [1018, 1018]
```

### Timestep 주입 방식 비교

```
π₀  (suffix level 처리):
  time_emb [4,1024] → repeat(s=32) → concat([action, time]) [4,32,2048]
                    → action_time_mlp_in (2048→1024) + SiLU
                    → action_time_mlp_out (1024→1024)
                    → action 토큰 자체에 time 정보가 혼합됨
  총 time 조건 주입: 1회 (suffix 생성 시)

π₀.₅ (layer level 처리, AdaRMSNorm):
  time_emb [4,1024] → time_mlp_in (1024→1024) + SiLU
                    → time_mlp_out (1024→1024) + SiLU
                    → adarms_cond [4,1024]
  매 Transformer layer에서:
    - Pre-Attention AdaRMSNorm: scale, shift, gate_attn
    - Pre-FFN AdaRMSNorm:       scale, shift, gate_ffn
  총 time 조건 주입: 18 × 2 + 1(final) = 37회
```

### AdaRMSNorm 작동 원리

```python
# 일반 RMSNorm (π₀):
scale = param([1024])          # 학습 가능, t와 무관
x_norm = x / rms(x) * (1 + scale)

# AdaRMSNorm (π₀.₅):
modulation = Dense(1024→3072)(time_emb)    # t에 따라 동적 생성
scale, shift, gate = split(modulation, 3)  # 각 [4, 1, 1024]

x_norm = x / rms(x) * (1 + scale) + shift  # t에 따른 feature 조정
residual = prev_x + output * gate           # t에 따른 residual 강도

# 직관:
# scale: 현재 t에서 어떤 feature를 강조할지
# shift: 현재 t에서 feature의 기준점을 어디로 옮길지
# gate:  현재 t에서 attention/FFN 결과를 얼마나 반영할지
```

---

## 📝 변경 이력

- 2026-02-15: 초안 작성
  - pi0_description.md를 기반으로 π₀.₅ 전용 문서 작성
  - π₀ vs π₀.₅ 차이점 상세 분석
  - AdaRMSNorm 동작 원리 상세 설명
  - State 텍스트 직렬화 방식 설명
  - 전체 shape 변화 추적 (π₀와 비교 포함)

---

**작성자**: AI Analysis
**프로젝트**: openpi (Physical Intelligence)
**버전**: 1.0
