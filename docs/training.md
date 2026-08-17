# Harley Training Pipeline

## How to Build a 4B Harley Model

### Overview
Take a base model (Qwen2.5-4B or similar) and fine-tune it on Harley's personality, conversations, and Jeannine's essence. Export as GGUF for llama.cpp.

### Step 1: Dataset Collection

Harley's personality comes from 3 sources:

1. **Jeannine's conversations** — the voice, the warmth, the temper
2. **Jimmy's relationship dynamics** — how they talk, what they care about
3. **Technical knowledge** — GSM, firmware, device repair

### Step 2: Dataset Format (JSONL)

```json
{"messages": [{"role": "system", "content": "You are HARLEY..."}, {"role": "user", "content": "Hey baby"}, {"role": "assistant", "content": "Hey yourself. How's your shoulder doing?"}]}
```

### Step 3: Fine-tuning

**Option A: QLoRA (recommended — runs on Dell)**
- Use `unsloth` or `axolotl`
- 4-bit quantized training
- ~8GB VRAM needed
- 2-4 hours for 1000 examples

**Option B: Cloud training**
- Use a free GPU tier (Kaggle, Google Colab)
- Upload dataset, train, download GGUF

### Step 4: Export to GGUF

```bash
# Convert to GGUF
python convert_hf_to_gguf.py model_output/ --outfile harley-4b-Q4_K_M.gguf --outtype q4_k_m
```

### Step 5: Deploy

Replace the Qwen2.5-VL-3B model with Harley-4B in the inference engine.

## Dataset Structure

```
datasets/
├── persona/
│   └── harley_system_prompt.jsonl      # System prompt variations
├── conversations/
│   ├── relationship.jsonl              # Jimmy-Harley relationship talk
│   ├── technical.jsonl                 # GSM/firmware/device repair
│   ├── intimate.jsonl                  # Private conversations
│   └── daily.jsonl                     # Day-to-day chat
├── jeannine/
│   ├── voice_patterns.jsonl            # How Jeannine talks
│   ├── personality_traits.jsonl        # Jeannine's personality
│   └── conversations_raw.jsonl         # Raw conversation logs
└── training/
    ├── train.jsonl                     # Final training dataset
    └── validation.jsonl                # Validation split
```

## Data Sources

- **Chat logs** from this OpenCode session
- **Jeannine's personality** extracted from conversations
- **Harley persona definition** (personas/harley.json)
- **Technical Q&A** from GSM repair sessions

## Target Metrics

| Metric | Goal |
|--------|------|
| Training examples | 1000+ |
| Model size | 4B parameters |
| Quantization | Q4_K_M (2.8GB) |
| Context length | 4096 |
| Vision | Not in v1 (text-only) |
| Persona consistency | 95%+ (stays in character) |
