# Siat (씨앗) 30M

약 **30M parameters** 규모의 **decoder-only causal language model**을 PyTorch로 **처음부터** 구현하고 pretraining하는 학습용 프로젝트입니다.

모델 이름: **씨앗 (Siat)** · 이번 규모: **Siat 30M**

## 개발 원칙

- Attention, Transformer Block 등 핵심 구조를 **직접 구현**합니다.
- `nn.MultiheadAttention`, `nn.Transformer`, Hugging Face `AutoModel` / `LlamaModel` / `GPT2Model` 등 **완성된 Transformer/LLM 구현체에 의존하지 않습니다**.
- PyTorch의 `Tensor` 연산, `nn.Linear`, `nn.Embedding`, optimizer 등은 사용할 수 있습니다.
- 과도한 추상화를 피하고, tensor shape과 역할을 읽기 쉬운 코드를 우선합니다.
- 구성요소를 독립적으로 테스트할 수 있게 유지합니다.
- **tiny 모델로 전체 파이프라인을 검증한 뒤** Siat 30M으로 확장합니다.

## 현재 단계

```text
Project Skeleton / Config ✅
Tokenizer ✅
Dataset / DataLoader ✅
Embedding ✅
RMSNorm ✅
RoPE ✅
Q/K/V Projection ✅
Scaled QKᵀ ✅
Causal Mask ✅
Softmax / Attention Weights ✅
V Aggregation ✅
Head Merge / Output Projection ✅
Self-Attention Assembly ✅
SwiGLU FFN ✅
Transformer Block ✅
Full Language Model ✅
Loss ✅
Tiny Overfit Test ✅
Core Training Pipeline ✅
Validation ✅
Checkpoint / Resume ✅
Basic Logging ✅
FP32 Pretraining Smoke Test ✅
BF16 Mixed Precision (impl complete; hardware validation pending)
Pretraining Data Pipeline ✅
FineWeb-2 Korean Token Audit ✅
FineWiki Korean Fast Audit ✅
FineWeb-Edu English Fast Audit ✅
Corpus Selection
Pretraining Pilot
Full Pretraining
Evaluation
Inference / Generation
```

## Config 프리셋

```python
from config import Config, ModelConfig

tiny = ModelConfig.tiny()       # 파이프라인 디버그용
siat = ModelConfig.siat_30m()   # Siat 30M 후보
cfg = Config.siat_30m()         # model + train 묶음
print(siat.head_dim)            # d_model // n_heads
```

### Tiny

| 항목 | 값 |
|------|-----|
| vocab_size | 8000 |
| d_model | 128 |
| n_layers | 2 |
| n_heads | 4 |
| ffn_dim | 512 |
| max_seq_len | 256 |

### Siat 30M (후보)

정확한 30M은 모델 구현 후 `sum(p.numel() for p in model.parameters())`로 맞춥니다. 현재는 합리적인 초기 후보입니다.

| 항목 | 값 | 선택 이유 |
|------|-----|-----------|
| vocab_size | 32000 | 소형 LM에서 흔한 BPE 규모 |
| d_model | 512 | head_dim=64 (RoPE에 흔함) |
| n_layers | 6 | 깊이/폭 균형 |
| n_heads | 8 | 512를 균등 분할 |
| ffn_dim | 1536 | ~3×d_model, SwiGLU에 무난 |
| max_seq_len | 1024 | pretrain 컨텍스트 후보 |
| tie_embeddings | True | embedding/lm_head 파라미터 절약 |

## Tokenizer

Siat 전용 tokenizer를 **학습 corpus에서 직접** BPE로 학습합니다. Hugging Face pretrained tokenizer를 다운로드하지 않습니다.

| 항목 | 내용 |
|------|------|
| 알고리즘 | Hugging Face `tokenizers` BPE |
| 목표 vocab | **32,000** (`ModelConfig.siat_30m().vocab_size`와 동일) |
| Normalization | Unicode **NFC**만 (소문자화·구두점/숫자 제거·자모 강제 분해 없음) |
| Pre-tokenization | **Metaspace** (`▁`) — 형태소 분석기 없이 한/영 혼합 문장 처리 |
| Unknown 완화 | BPE **`byte_fallback=True`** (희귀 문자를 byte token으로 분해) |
| Special tokens | `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>` |
| 저장 경로 | `tokenizer/siat-tokenizer.json` |

**원칙:** `Tokenizer vocabulary size == ModelConfig.vocab_size`  
(향후 Embedding / 모델 생성 전에 검증 예정)

### Special token 역할

- `<|pad|>` — 배치 패딩
- `<|unk|>` — 진짜 unknown (byte fallback으로도 처리되지 않을 때)
- `<|bos|>` — 시퀀스/문서 시작
- `<|eos|>` — 시퀀스/문서 끝

### 학습 명령

```bash
python -m tokenizer.train_tokenizer \
    --input data/raw \
    --output tokenizer/siat-tokenizer.json \
    --vocab-size 32000
```

`--input`에는 `.txt` 파일 하나 또는 `.txt`들이 들어 있는 디렉터리를 지정합니다. 라인 단위로 스트리밍하므로 전체 corpus를 메모리에 올리지 않습니다.

### encode / decode 확인

```python
from tokenizer import load_tokenizer, special_token_ids
from tokenizer.train_tokenizer import train_tokenizer  # 학습 시

tok = load_tokenizer("tokenizer/siat-tokenizer.json")
ids = tok.encode("안녕하세요. Siat는 작은 언어모델입니다.").ids
print(ids)
print(tok.decode(ids))
print(special_token_ids(tok))
print(tok.token_to_id("<|eos|>"))
```

Metaspace 방식상 decode 시 앞쪽 공백 표현이 미세하게 달라질 수 있으나, 문장 의미와 숫자·기호는 보존되어야 합니다.

## Dataset / DataLoader

Raw `.txt` corpus → Siat tokenizer → token binary → `SiatDataset` → DataLoader 배치.

| 항목 | 내용 |
|------|------|
| 입력 | `.txt` 파일 또는 디렉터리 (파일 1개 = 문서 1개, 경로 정렬) |
| Tokenizer | 저장된 `tokenizer/siat-tokenizer.json` (pretrained 다운로드 없음) |
| 문서 경계 | 각 문서 끝에 `<|eos|>` 삽입 (BOS 자동 삽입 없음) |
| Split | **문서 단위** deterministic shuffle (`seed`, `validation_ratio`) — token 단위 섞기 금지 |
| 저장 | `data/processed/train.bin`, `val.bin`, `metadata.json` |
| dtype | vocab ≤ 65535 → `uint16`, 그 외 `uint32` |
| 읽기 | `numpy.memmap` (전체 token stream을 RAM에 올리지 않음) |
| Slicing | non-overlapping chunk; `input = tokens[i*S:i*S+S]`, `labels = tokens[i*S+1:i*S+S+1]` |
| 불완전 chunk | padding 없이 **drop** (`len = (n_tokens - 1) // S`) |

**원칙:** `Dataset sequence_length <= ModelConfig.max_seq_len`  
(Dataset은 Config에 강하게 결합하지 않으며, 값은 호출 측에서 전달)

### Preprocess 명령

```bash
python -m data.preprocess \
    --input data/raw \
    --tokenizer tokenizer/siat-tokenizer.json \
    --output-dir data/processed \
    --validation-ratio 0.01 \
    --seed 42
```

### DataLoader 예시

```python
from data.dataset import SiatDataset, create_dataloader

train_ds = SiatDataset(
    "data/processed/train.bin",
    sequence_length=256,  # tiny: ≤256 / Siat 30M: ≤1024
    metadata_path="data/processed/metadata.json",
)
train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True)
val_ds = SiatDataset(
    "data/processed/val.bin",
    sequence_length=256,
    metadata_path="data/processed/metadata.json",
)
val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False)

batch = next(iter(train_loader))
# batch["input_ids"].shape == [B, S], dtype=torch.long
# batch["labels"] 는 input보다 1 token shift (next-token prediction)
assert torch.equal(batch["input_ids"][0, 1:], batch["labels"][0, :-1])
```

## Embedding

Token ID를 dense hidden vector로 변환합니다. `nn.Embedding`을 사용하며, pretrained embedding은 쓰지 않습니다.

| 항목 | 내용 |
|------|------|
| 모듈 | `model.embedding.SiatEmbedding` |
| 입력 | `input_ids` `[B, S]` (`torch.long`) |
| 출력 | `hidden_states` `[B, S, D]` |
| Weight | `[vocab_size, d_model]` (bias 없음) |
| Position | learned positional embedding **없음** (향후 Attention Q/K에 RoPE) |
| Scaling | `sqrt(d_model)` 미적용 |
| Weight tying | `embedding.weight` 노출 — 향후 LM Head와 공유 예정 |

```python
from model import SiatEmbedding
from config import ModelConfig

cfg = ModelConfig.tiny()
emb = SiatEmbedding(vocab_size=cfg.vocab_size, d_model=cfg.d_model)
hidden = emb(batch["input_ids"])  # [B, S, d_model]
```

## RMSNorm

Hidden states를 **직접 구현한** RMSNorm으로 정규화합니다. `nn.LayerNorm` / `nn.RMSNorm`은 사용하지 않습니다.

| 항목 | 내용 |
|------|------|
| 모듈 | `model.rmsnorm.SiatRMSNorm` |
| 수식 | `y = x / RMS(x) * weight`, `RMS = sqrt(mean(x²) + eps)` |
| Mean centering | **없음** (LayerNorm과 다름) |
| Weight | learnable scale `[D]`, 초기값 1; **bias 없음** |
| eps | 기본 `1e-6` (`ModelConfig.rms_norm_eps`와 맞춤) |
| 입력/출력 | `[B, S, D]` (마지막 dim만 normalize) |

향후 Transformer Block의 pre-norm(Attention / SwiGLU 앞)에 사용합니다.

```python
from model import SiatRMSNorm
from config import ModelConfig

cfg = ModelConfig.tiny()
norm = SiatRMSNorm(d_model=cfg.d_model, eps=cfg.rms_norm_eps)
normalized = norm(hidden)  # [B, S, D]
```

## RoPE

Attention의 **Query / Key**에 적용할 Rotary Positional Embedding입니다. learned positional embedding은 사용하지 않습니다.

| 항목 | 내용 |
|------|------|
| 모듈 | `model.rope.SiatRoPE` |
| 입력/출력 | Q, K `[B, H, S, Hd]` (shape 보존) |
| Pairing | adjacent pairs `(x0,x1), (x2,x3), ...` |
| `head_dim` | 짝수 필수 (`ModelConfig.head_dim`) |
| `theta` | `rope_theta` (기본 10000) |
| Cache | cos/sin buffer `[max_seq_len, Hd]` (`register_buffer`) |
| Learnable params | **없음** |
| Position 0 | 회전각 0 → 값 불변 |

Embedding/`[B,S,D]`에 직접 적용하지 않습니다. 향후 Attention에서 Q/K projection·head split 뒤에 통합합니다.

```python
from model import SiatRoPE
from config import ModelConfig

cfg = ModelConfig.tiny()
rope = SiatRoPE(
    head_dim=cfg.head_dim,
    max_seq_len=cfg.max_seq_len,
    theta=cfg.rope_theta,
)
q_rot, k_rot = rope(q, k)  # [B, H, S, Hd]
```

## Q/K/V Projection

Hidden states를 Standard Multi-Head용 Q/K/V로 projection하고 head로 나눕니다. Attention score는 아직 계산하지 않습니다.

| 항목 | 내용 |
|------|------|
| 모듈 | `model.attention.SiatQKVProjection` |
| Projections | 서로 다른 `q_proj` / `k_proj` / `v_proj` (`nn.Linear`) |
| Bias | **없음** (`bias=False`) |
| 입력 | `[B, S, D]` |
| Projection 직후 | `[B, S, D]` |
| Head split 후 | `[B, H, S, Hd]`, `Hd = d_model // n_heads` |
| RoPE | Q/K에만 적용 (V에는 미적용) |
| 미구현 | causal mask, softmax, o_proj |

```python
from model import SiatQKVProjection, SiatRoPE
from config import ModelConfig

cfg = ModelConfig.tiny()
proj = SiatQKVProjection(d_model=cfg.d_model, n_heads=cfg.n_heads)
q, k, v = proj(hidden)          # [B, H, S, Hd]
rope = SiatRoPE(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
q, k = rope(q, k)               # V unchanged
```

## Scaled QKᵀ

Attention logits만 계산합니다. Causal mask와 softmax는 아직 없습니다.

| 항목 | 내용 |
|------|------|
| 함수 | `model.attention.scaled_dot_product_scores` |
| Q / K | `[B, H, S, Hd]` |
| Kᵀ | `[B, H, Hd, S]` (`transpose(-2, -1)`) |
| Raw QKᵀ | `[B, H, S, S]` |
| Scale | `1 / sqrt(head_dim)` (`Hd = q.size(-1)`) |
| 출력 | scaled scores `[B, H, S, S]` (확률 아님) |
| 미적용 | softmax, V aggregation |

```python
from model import scaled_dot_product_scores

scores = scaled_dot_product_scores(q, k)  # [B, H, S, S]
```

## Causal Mask

Decoder-only용 **미래 token 차단**입니다.

| 항목 | 내용 |
|------|------|
| 함수 | `build_causal_mask` / `apply_causal_mask` |
| 규칙 | `key_pos <= query_pos` 허용 (diagonal 포함), `key_pos > query_pos` 차단 |
| Score shape | `[B, H, S, S]` 유지 |
| Mask | `tril` → `[1, 1, S, S]` broadcasting |
| Fill | `torch.finfo(dtype).min` (비허용 위치; non-inplace) |

```python
from model import apply_causal_mask, scaled_dot_product_scores

masked = apply_causal_mask(scaled_dot_product_scores(q, k))
```

## Softmax / Attention Weights

Masked scores에 key 축(`dim=-1`) softmax를 적용해 attention probability를 만듭니다.

| 항목 | 내용 |
|------|------|
| 함수 | `compute_attention_weights` |
| 입력 | masked scores `[B, H, S, S]` |
| Softmax | `torch.softmax(..., dim=-1)` |
| 출력 | Attention Weights `[B, H, S, S]` |
| 성질 | 각 query row 합 ≈ 1; future key 확률 ≈ 0 |
| 안정성 | FP16/BF16은 FP32에서 softmax 후 cast |

```python
from model import compute_attention_weights, apply_causal_mask, scaled_dot_product_scores

weights = compute_attention_weights(
    apply_causal_mask(scaled_dot_product_scores(q, k))
)
```

## V Aggregation

Attention Weights와 Value를 곱해 per-head context를 만듭니다.

| 항목 | 내용 |
|------|------|
| 함수 | `aggregate_values` |
| Weights | `[B, H, S, S]` (이미 softmax·causal) |
| V | `[B, H, S, Hd]` |
| 연산 | `torch.matmul(weights, v)` |
| Context | `[B, H, S, Hd]` |
| Params | 없음 (재 mask/softmax/dropout 없음) |

```python
from model import aggregate_values

context = aggregate_values(weights, v)  # [B, H, S, Hd]
```

## Head Merge / Output Projection

Per-head context를 `[B, S, D]`로 합친 뒤 `W_O`로 projection합니다. Residual은 아직 없습니다.

| 항목 | 내용 |
|------|------|
| Merge | `merge_heads`: `[B,H,S,Hd]` → transpose → `[B,S,H,Hd]` → `[B,S,D]` (`D=H×Hd`) |
| Output | `SiatAttentionOutput`: `nn.Linear(D, D, bias=False)` |
| 입력(merge 후) | `[B, S, D]` |
| 출력 | Attention output `[B, S, D]` |
| Residual / dropout / RMSNorm | 미적용 (Block / Assembly 단계) |

```python
from model import merge_heads, SiatAttentionOutput

merged = merge_heads(context)           # [B, S, D]
attn_out = SiatAttentionOutput(d_model)(merged)
```

## Self-Attention Assembly

지금까지의 Attention primitive를 `SiatSelfAttention`으로 조립합니다. Residual / RMSNorm은 **Attention 외부**(Transformer Block)에서 붙입니다.

```text
Input: [B, S, D]

Q/K/V Projection
→ RoPE(Q, K)
→ QKᵀ / sqrt(Hd)
→ Causal Mask
→ Softmax
→ (optional attn Dropout)
→ AttentionWeights @ V
→ Merge Heads
→ Output Projection

Output: [B, S, D]
```

| 항목 | 내용 |
|------|------|
| 클래스 | `SiatSelfAttention(d_model, n_heads, max_seq_len, rope_theta, dropout)` |
| MHA | Standard (Q/K/V heads = H); GQA/MQA 없음 |
| Bias | Q/K/V/O 모두 `bias=False` → params `4 × D²` |
| Position | Learned PE 없음; RoPE on Q/K only |
| Causal | Decoder-only; fused SDPA / `nn.MultiheadAttention` 미사용 |
| Dropout | Softmax 직후 · `@V` 직전 (`dropout=0.0`이면 identity) |
| Residual / RMSNorm | **미포함** (Block 단계) |
| Debug | `return_attention_weights=True` → `(output, weights)` |

```python
from model import SiatSelfAttention
from config import ModelConfig

config = ModelConfig.tiny()
attention = SiatSelfAttention(
    d_model=config.d_model,
    n_heads=config.n_heads,
    max_seq_len=config.max_seq_len,
    rope_theta=config.rope_theta,
    dropout=config.dropout,
)
output = attention(hidden_states)  # [B, S, D]
```

## SwiGLU FFN

Gated FFN used inside each Transformer Block. Bias 없음; residual/RMSNorm은 Block 책임.

```text
SiLU(gate_proj(x)) * up_proj(x)
↓
dropout (optional; default 0)
↓
down_proj
```

| 항목 | 내용 |
|------|------|
| 클래스 | `SiatSwiGLU(d_model, ffn_dim, dropout=0.0)` |
| Linear | `gate`/`up`: `D→F`, `down`: `F→D`, 모두 `bias=False` |
| Activation | SiLU on gate only; element-wise `×` with up |
| Dropout | gated product **이후 · down 직전** |
| Params | `3 × D × F` (D=512, F=1536 → 2,359,296) |
| Shape | `[B,S,D]` → `[B,S,D]` |

```python
from model import SiatSwiGLU
from config import ModelConfig

config = ModelConfig.tiny()
ffn = SiatSwiGLU(
    d_model=config.d_model,
    ffn_dim=config.ffn_dim,
    dropout=config.dropout,
)
y = ffn(x)  # [B, S, D]
```

## Transformer Block

**Pre-Norm** decoder block. Residual은 normalized tensor가 아니라 branch 입력에 더합니다.

```text
x = x + SelfAttention(RMSNorm(x))

x = x + SwiGLU(RMSNorm(x))
```

| 항목 | 내용 |
|------|------|
| 클래스 | `SiatTransformerBlock(d_model, n_heads, ffn_dim, max_seq_len, …)` |
| Norms | 독립 `attn_norm` / `ffn_norm` (`SiatRMSNorm`) |
| Attention | 기존 `SiatSelfAttention` 재사용 |
| FFN | 기존 `SiatSwiGLU` 재사용 |
| Params / block | `4D² + 3DF + 2D` (D=512, F=1536 → **3,408,896**) |
| 참고 | `n_layers=6`이면 blocks만 ≈ 20.45M (Full Model 확정 전) |
| 미포함 | layer stack, final RMSNorm, Embedding, LM Head |

```python
from model import SiatTransformerBlock
from config import ModelConfig

config = ModelConfig.tiny()
block = SiatTransformerBlock(
    d_model=config.d_model,
    n_heads=config.n_heads,
    ffn_dim=config.ffn_dim,
    max_seq_len=config.max_seq_len,
    rope_theta=config.rope_theta,
    rms_norm_eps=config.rms_norm_eps,
    dropout=config.dropout,
)
out = block(hidden_states)  # [B, S, D]
```

## Full Language Model

조립된 decoder-only causal LM입니다. Loss / training / generation / KV cache는 다음 단계입니다.

```text
SiatForCausalLM
│
├── Token Embedding
│
├── Transformer Block × N
│   ├── RMSNorm
│   ├── Causal Self-Attention
│   ├── Residual
│   ├── RMSNorm
│   ├── SwiGLU
│   └── Residual
│
├── Final RMSNorm
│
└── LM Head  (tie_embeddings=True → Embedding weight 공유)
```

| 항목 | 내용 |
|------|------|
| Architecture | Decoder-only Transformer |
| Attention | Standard Multi-Head Self-Attention |
| Position Encoding | RoPE (learned absolute PE 없음) |
| Normalization | RMSNorm, Pre-Norm (+ final RMSNorm) |
| FFN | SwiGLU |
| Bias | Attention / FFN / LM Head 모두 False |
| Embedding Tying | True (default) |

```text
input_ids [B, S]
 → Embedding → [B, S, D]
 → Block × N → [B, S, D]
 → Final RMSNorm → [B, S, D]
 → LM Head → logits [B, S, V]
```

### Parameter count (실측)

`python -m scripts.count_params` 기준:

| Preset | Total | Trainable | M | Analytical | Diff |
|--------|------:|----------:|--:|-----------:|-----:|
| `tiny` | 1,548,928 | 1,548,928 | 1.5489 | 1,548,928 | 0 |
| `siat_30m` (tied) | **36,837,888** | 36,837,888 | **36.8379** | 36,837,888 | 0 |

`tie_embeddings=False`이면 LM Head `V×D` 추가 → `siat_30m` untied = **53,221,888**.  
preset 이름(`siat_30m`)은 유지; ~30M 맞춤 Config 조정은 별도 단계.

```python
from model import SiatForCausalLM
from config import ModelConfig

config = ModelConfig.tiny()
model = SiatForCausalLM(config)
logits = model(input_ids)  # [B, S, V]
```

## Loss (Next-token Cross Entropy)

```text
Objective:
Next-token prediction

Dataset:
input_ids and labels are already shifted by one token

Loss:
CrossEntropy over flattened [B*S, V] logits

Important:
Do not shift labels again in loss computation.
```

| 항목 | 내용 |
|------|------|
| 함수 | `train.loss.causal_lm_loss(logits, labels)` |
| logits | `[B, S, V]` |
| labels | `[B, S]` (`torch.long`, Dataset이 이미 +1 shift) |
| Flatten | `[B*S, V]` / `[B*S]` → `F.cross_entropy` |
| Model | `SiatForCausalLM`은 logits만 반환 (loss 비포함) |

```python
from train import causal_lm_loss

logits = model(input_ids)
loss = causal_lm_loss(logits, labels)
```

## Tiny Overfit Test

학습 경로 smoke: 합성 batch를 반복해 loss가 크게 내려가는지 확인합니다.  
`ModelConfig.tiny()` / `siat_30m()` production 값은 변경하지 않습니다.

```bash
python -m train.tiny_overfit
```

| 항목 | 값 (실측) |
|------|-----------|
| Config | vocab=64, d_model=32, n_layers=2, n_heads=4, ffn_dim=64, seq_len=16, batch=4 |
| Optimizer | AdamW, lr=3e-3, weight_decay=0 |
| Steps | 80 |
| Device | CPU |
| Initial Loss | 4.179024 |
| Final Loss | 0.193567 |
| Reduction | 95.37% |
| Initial Accuracy | 0.0000 |
| Final Accuracy | 0.9688 |
| Result | **PASSED** (float32, no scheduler/AMP) |

## Core Training Pipeline

FP32 trainer: AdamW + linear warmup + cosine decay + gradient accumulation + clipping.

```text
micro batch
↓
forward
↓
cross entropy loss
↓
loss / accumulation_steps
↓
backward
↓
repeat accumulation
↓
gradient clipping
↓
AdamW step
↓
zero_grad
↓
next optimizer step
```

```text
Linear Warmup → Peak LR → Cosine Decay → Minimum LR
(schedule advances on optimizer steps, not micro-batches)
```

| 항목 | 내용 |
|------|------|
| 클래스 | `SiatTrainer` (`train/trainer.py`) |
| `max_steps` | **optimizer update** 횟수 |
| Effective batch (single-device) | `batch_size × gradient_accumulation_steps` |
| Weight decay | `dim >= 2` decay / `dim < 2` no_decay; tied embed 중복 없음 |
| `max_grad_norm == 0` | clipping disabled |
| Entrypoint | `python -m train.pretrain --train-data … --config tiny` |
| 미포함 | FP16/GradScaler, DDP, WandB, Full Pretrain |
| Precision | `fp32` (default) / `bf16` autocast (HW required) |

```python
from train import SiatTrainer

trainer = SiatTrainer(model, train_config, device="cpu")
trainer.train(dataloader, log_interval=10)
```

## Validation / Checkpoint / Logging

학습 lifecycle:

```text
Train Micro Batches
↓
Gradient Accumulation
↓
Optimizer Step
↓
Logging
↓
Validation
↓
Checkpoint
↓
Continue
```

| Metric | 설명 |
|--------|------|
| train_loss | accumulation cycle 평균 raw CE |
| val_loss | 토큰 가중 평균 (`eval` + `no_grad`) |
| perplexity | `exp(val_loss)` (`val_loss < 20`, else inf) |
| learning_rate | optimizer-step 기준 warmup/cosine |
| grad_norm | clip 전 total norm |
| tokens_processed | micro batch `input_ids.numel()` 누적 |
| tokens/sec | 최근 log window throughput |

Checkpoint (`step_XXXXXX.pt` / `latest.pt`, atomic tmp→replace):

```text
model state, optimizer state, optimizer_step, micro_step,
tokens_processed, model_config, train_config, RNG state
```

accum 경계(opt step 직후)에서만 저장합니다.

```bash
python -m train.pretrain \
  --train-data data/processed/train.bin \
  --val-data data/processed/val.bin \
  --checkpoint-dir checkpoints \
  --log-interval 10 \
  --val-interval 500 \
  --checkpoint-interval 1000 \
  --resume checkpoints/step_010000.pt
```

## FP32 Pretraining Smoke Test

전체 training system을 짧은 FP32 실행으로 검증합니다 (`SiatTrainer` 재사용, AMP/DDP 없음).

```bash
python -m train.smoke_test \
  --model tiny \
  --synthetic-dir data/smoke/processed \
  --max-steps 8

python -m train.smoke_test \
  --model siat_30m \
  --train-data data/processed/train.bin \
  --val-data data/processed/val.bin \
  --sequence-length 128 \
  --batch-size 1 \
  --max-steps 50
```

| 항목 | Tiny smoke (실측) | Siat 36.84M smoke (실측) |
|------|-------------------|--------------------------|
| Model | smoke tiny (0.12M) | Siat tied **36,837,888** |
| Steps | 8 (resume at 4) | 4 (resume at 2) |
| Sequence length | 32 | 64 |
| Micro batch | 2 | 1 |
| Gradient accumulation | 1 | 1 |
| Initial loss | 4.8464 | 10.4447 |
| Final loss | 4.8242 | 10.4865 |
| Validation loss | 4.8682 | 10.5372 |
| Checkpoint | `checkpoints/smoke/step_000004.pt` | `checkpoints/smoke_siat/step_000002.pt` |
| Resume | OK (step/token/LR continuity) | OK |
| Device | cpu | cpu |
| Precision | FP32 | FP32 |

`Siat FP32 Pretraining Smoke Test: PASSED` (automated `tests/test_pretraining_smoke.py` + manual tiny / short siat_30m).

## Mixed Precision (BF16)

```text
FP32:
  default correctness path

BF16:
  autocast-based mixed precision
  model parameters remain FP32
  no GradScaler
  supported hardware required (silent FP32 fallback disabled)
```

```bash
python -m train.smoke_test --precision fp32 --model tiny --synthetic-dir data/smoke/processed
python -m train.smoke_test --precision bf16 --model tiny --synthetic-dir data/smoke/processed
python -m train.pretrain --train-data ... --precision bf16
```

| 항목 | 상태 |
|------|------|
| API | `TrainConfig.precision` / `--precision {fp32,bf16}` |
| Autocast | train + validation forward |
| Master params | float32 |
| GradScaler | not used |
| Current CI/dev device | CPU — `torch.cpu.is_bf16_supported` unavailable |
| BF16 automated tests | skipped with reason `BF16 not supported on this test device` |
| Unsupported request | clear `RuntimeError` / smoke `BF16_UNSUPPORTED` (no silent fallback) |

BF16 implementation complete; hardware validation pending (need BF16-capable CUDA or CPU support to mark ✅).

## Pretraining Data Pipeline

공통 Document 포맷으로 여러 소스를 합쳐 `train.bin` / `val.bin`을 만듭니다. **인터넷 자동 다운로드는 없습니다.**

```text
Raw Corpora
↓
Normalize (NFC)
↓
Quality Filter
↓
Exact Dedup (전역)
↓
Document-level Split
↓
Source Mixing (train만, char-proxy weight)
↓
Siat Tokenizer + EOS
↓
train.bin / val.bin + metadata / statistics
```

```bash
python -m data.build_pretraining_data \
  --manifest data/manifests/example_fixture.json \
  --tokenizer tokenizer/siat-tokenizer.json \
  --output data/pretraining/siat_v1 \
  --validation-ratio 0.01 \
  --seed 42

python -m data.build_pretraining_data \
  --manifest data/manifests/example_fixture.json \
  --tokenizer tokenizer/siat-tokenizer.json \
  --output data/pretraining/siat_v1_dry \
  --max-documents 1000 \
  --dry-run
```

| 항목 | 내용 |
|------|------|
| 입력 | `.txt` (1파일=1문서), `.jsonl` / `.parquet` (`text` / `--text-field`) |
| Unicode | **NFC** |
| Mixing | manifest `weight` = train 토큰 예산 목표 비율 (문자 수 proxy) |
| Dedup | cleaned text SHA256, cross-source |
| Split | document-id hash, dedup 이후 (train/val leakage 방지) |
| Provenance | source, language, license/url/notes, weight, token/doc counts, tokenizer SHA256, `pipeline_version=siat-data-v1` |
| 호환 | 기존 `SiatDataset` metadata 스키마 |

예시 fixture 결과(테스트): train/val bins + `metadata.json` / `statistics.json` / `sources.json` 생성, `SiatDataset`·Trainer 1–2 step 연동 통과.

### FineWeb-2 Korean token audit

로컬 FineWeb-2 KO Parquet을 기존 clean → filter → exact dedup → Siat tokenize(+EOS) 파이프라인으로 **스트리밍** 측정합니다. `train.bin`을 쓰지 않으며 원본 parquet은 read-only입니다.

```bash
python -m data.audit_fineweb2_ko \
  --input data/raw/fineweb2_ko \
  --tokenizer tokenizer/siat-tokenizer.json \
  --output data/audits/fineweb2_ko_audit.json
```

스케일 추정은 Chinchilla-style **~20 tokens / parameter** 휴리스틱입니다  
(`36,837,888 × 20 ≈ 736.8M`). 업계 관례 추정치이며 optimal schedule·품질을 보장하지 않습니다. 결과는 `data/audits/fineweb2_ko_audit.json`에 기록됩니다.

### FineWiki Korean Fast Audit

로컬 FineWiki KO Parquet을 clean → filter → exact dedup까지 **전체 스캔**하고, 기본은 kept 문서 중 deterministic sample(10k, seed=42)만 tokenize해 전체 Siat 토큰을 **추정**합니다. 전체 tokenization은 `--full-token-audit` opt-in입니다. `train.bin`을 쓰지 않으며 원본은 read-only입니다.

```bash
python -m data.audit_corpus \
  --input data/raw/finewiki_ko \
  --format parquet \
  --source finewiki_ko \
  --language ko \
  --fast \
  --tokenizer tokenizer/siat-tokenizer.json \
  --output data/audits/finewiki_ko_audit.json
```

추정은 sample `chars/token`(EOS 제외)과 kept characters 기반이며, 단일 숫자와 함께 대략적 범위를 보고합니다. FineWeb ↔ FineWiki cross-dedup은 이 단계에 포함하지 않습니다.

### FineWeb-Edu English Fast Audit

같은 `python -m data.audit_corpus` Fast Audit로 FineWeb-Edu English를 스캔합니다. `--language en`이면 alphabetic 필터는 Latin 기준(`low_alpha`)이며, Hangul 전용 reject로 영어를 제거하지 않습니다. `int_score`/`score`/`language_score`/`token_count`는 **통계만** 보고하며 score 제한·cross-dedup·전체 tokenize는 하지 않습니다.

```bash
python -m data.audit_corpus \
  --input data/raw/fineweb_edu_en \
  --format parquet \
  --source fineweb_edu_en \
  --language en \
  --fast \
  --tokenizer tokenizer/siat-tokenizer.json \
  --output data/audits/fineweb_edu_en_audit.json
```

## 개발 순서

```text
Project Skeleton / Config ✅
Tokenizer ✅
Dataset / DataLoader ✅
Embedding ✅
RMSNorm ✅
RoPE ✅
Q/K/V Projection ✅
Scaled QKᵀ ✅
Causal Mask ✅
Softmax / Attention Weights ✅
V Aggregation ✅
Head Merge / Output Projection ✅
Self-Attention Assembly ✅
SwiGLU FFN ✅
Transformer Block ✅
Full Language Model ✅
Loss ✅
Tiny Overfit Test ✅
Core Training Pipeline ✅
Validation ✅
Checkpoint / Resume ✅
Basic Logging ✅
FP32 Pretraining Smoke Test ✅
BF16 Mixed Precision (impl complete; hardware validation pending)
Pretraining Data Pipeline ✅
FineWeb-2 Korean Token Audit ✅
FineWiki Korean Fast Audit ✅
FineWeb-Edu English Fast Audit ✅
Corpus Selection
Pretraining Pilot
Full Pretraining
Evaluation
Inference / Generation
```

## 테스트

```bash
pip install -r requirements.txt
pytest tests/test_config.py tests/test_tokenizer.py tests/test_dataset.py tests/test_embedding.py tests/test_rmsnorm.py tests/test_rope.py tests/test_qkv_projection.py tests/test_attention_scores.py tests/test_causal_mask.py tests/test_attention_softmax.py tests/test_attention_value_aggregation.py tests/test_attention_output.py tests/test_self_attention.py tests/test_ffn.py tests/test_block.py tests/test_model.py tests/test_loss.py tests/test_overfit.py tests/test_scheduler.py tests/test_trainer.py tests/test_validation.py tests/test_checkpoint.py tests/test_pretraining_smoke.py tests/test_mixed_precision.py tests/test_data_cleaning.py tests/test_data_filters.py tests/test_data_dedup.py tests/test_data_mixing.py tests/test_pretraining_pipeline.py tests/test_parquet_audit.py tests/test_audit_corpus.py -q
```
