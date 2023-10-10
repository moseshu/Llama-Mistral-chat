#!/bin/bash
lora_model=chekpoint
python -u gradio_web.py --base_model Moses25/Llama2-MosesLM-7b-chat \
    --lora_model ${lora_model} \
	--alpha 1 \
	--post_host 0.0.0.0 \
	--port 80
