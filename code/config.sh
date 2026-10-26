#!/bin/bash

# target_model is set by each domain's run.sh before sourcing this file
model_path=""

export api_base="https://api.openai.com/v1"
export api_keys="sk-your-api-key-1,sk-your-api-key-2"

case "$target_model" in
  "gpt4o")
    model_name="gpt-4o"
    echo ">>> Selected Model: GPT-4o"
    max_tokens=4096
    temperature=0.7  
    ;;
  "gemini2.5pro")
    model_name="gemini-2.5-pro"
    echo ">>> Selected Model: Gemini 2.5 Pro"
    max_tokens=4096
    temperature=0.7  
    ;;
  "gemini3pro")
    model_name="gemini-3-pro-preview-thinking"
    echo ">>> Selected Model: Gemini 3 Pro (Thinking)"
    max_tokens=4096
    temperature=0.7 
    ;;
  "deepseekr1")
    model_name="deepseek-reasoner"
    echo ">>> Selected Model: DeepSeek R1 (deepseek-reasoner)"
    max_tokens=4096
    temperature=0.7  
    ;;
  "qwen2.5_7b")
    model_name="qwen2.5-7b-instruct"
    echo ">>> Selected Model: Qwen2.5 7B (instruct)"
    model_path="/path/to/Qwen2.5-7B-Instruct"
    max_tokens=4096
    temperature=0.7
    ;;
  "qwen2.5_14b")
    model_name="qwen2.5-14b-instruct"
    echo ">>> Selected Model: Qwen2.5 14B (instruct)"
    model_path="/path/to/Qwen2.5-14B-Instruct"
    max_tokens=4096
    temperature=0.7
    ;;
  "deepseek_r1_qwen14b")
    model_name="deepseek-r1-qwen14b"
    echo ">>> Selected Model: DeepSeek R1 Qwen14B"
    model_path="/path/to/DeepSeek-R1-Distill-Qwen-14B"
    max_tokens=4096
    temperature=0.7
    ;;
esac
