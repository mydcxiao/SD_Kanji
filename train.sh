export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="data/kanji_images"
export VAL_PROMPT="Gundam"
# export HUB_TOKEN="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"
# export HUB_MODEL_ID="sd_kanji"
export OUTPUT_DIR="/lab/tmpig7b/u/yxiao-data/sd_kanji"

accelerate launch --mixed_precision="fp16"  train_kanji.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=40000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="$OUTPUT_DIR" \
  --validation_prompt="$VAL_PROMPT" \
  --report_to="wandb" \
  --tracker_project_name="sd_kanji" \
  --push_to_hub \
  --checkpointing_steps=5000 \
  --validation_epochs=1 \
#   --enable_xformers_memory_efficient_attention \
#   --hub_token="$HUB_TOKEN" \
#   --hub_model_id="$HUB_MODEL_ID" \