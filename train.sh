export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="data/kanji_images"
export VAL_PROMPT="abandon"
export OUTPUT_DIR="/lab/tmpig7b/u/yxiao-data/sd_kanji"
# export HUB_TOKEN="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"
# export HUB_MODEL_ID="sd_kanji"
# export RESUME="/lab/tmpig7b/u/yxiao-data/sd_kanji/checkpoint-40000"

accelerate launch --mixed_precision="fp16"  train_kanji.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=200000 \
  --snr_gamma=5.0 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="$OUTPUT_DIR" \
  --validation_prompt="$VAL_PROMPT" \
  --report_to="wandb" \
  --tracker_project_name="sd_kanji" \
  --checkpointing_steps=50000 \
  --validation_epochs=1 \
  --use_8bit_adam \
  --from_scratch \
  # --resume_from_checkpoint="latest" \
  # --push_to_hub \
  # --enable_xformers_memory_efficient_attention \
#   --hub_token="$HUB_TOKEN" \
#   --hub_model_id="$HUB_MODEL_ID" \