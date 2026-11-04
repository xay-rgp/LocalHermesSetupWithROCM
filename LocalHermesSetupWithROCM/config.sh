# Edit this file, then run: bash setup.sh
# Bash syntax: one quoted Ollama model name per line. These names are also
# used in the Ollama launcher. Old model downloads are NOT deleted.
MODELS=(
  "qwen3-coder:30b"
  "huihui_ai/Qwen3.8-abliterated:27b"
)
DEFAULT_MODEL="${MODELS[0]}"

# Shared Ollama service allocation and Hermes model.context_length.
# Keep these equal so Hermes compacts before Ollama truncates user messages.
# 64K balances agent startup/tool context against VRAM on a 16 GB card.
# Larger values increase cache memory and CPU offload.
CONTEXT_LENGTH=65536
KEEP_ALIVE="10m"
FLASH_ATTENTION=1
KV_CACHE_TYPE="q8_0"  # roughly half the f16 KV memory; use f16 if quality regresses

# Hermes release channel: stable releases or the development main branch.
HERMES_CHANNEL="stable"
