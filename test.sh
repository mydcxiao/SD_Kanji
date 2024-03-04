export CUDA_VISIBLE_DEVICES=0
export PROMPT="Shinjuku"
export MODEL_PATH="/lab/tmpig7b/u/yxiao-data/sd_kanji"
export NUM_IMAGES=20
export OUTPUT_DIR="outputs"
export CHECKPOINT=250000

python  test_kanji.py \
  --model_path="$MODEL_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --prompt="$PROMPT" \
  --num_images=$NUM_IMAGES \
  --checkpoint=$CHECKPOINT