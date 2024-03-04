export CUDA_VISIBLE_DEVICES=0
export PROMPT="abandon"
export MODEL_PATH="/lab/tmpig7b/u/yxiao-data/sd_kanji"
export NUM_IMAGES=10
export OUTPUT_DIR="outputs"
export CHECKPOINT=300000

python  test_kanji.py \
  --model_path="$MODEL_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --prompt="$PROMPT" \
  --num_images=$NUM_IMAGES \
#   --checkpoint=$CHECKPOINT