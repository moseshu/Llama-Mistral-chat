#!/bin/bash
lora_model=checkpoint/adapter_model
python -u webui_gradio.py --base_model Llama-2-7b-chat-hf \
    --lora_model ${lora_model} \
	--alpha 1 \
	--post_host 0.0.0.0 \
	--port 80
