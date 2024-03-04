# Stable Diffusion Hallucinating Kanji

Finetune/train a stable diffusion model using Kanji image dictionary to let it generate novel Kanjis.

Feel free to get my own [trained weights](https://huggingface.co/mydcxiao/SD_Kanji) with the same configuration as [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4).

## Dataset

KANJIDIC2 is a Kanji dictionary using a combination of .xml and .svg images to describe Kanjis.
Here are links to download the dataset.
- [KANJIDIC2](https://www.edrdg.org/kanjidic/kanjidic2.xml.gz)
- [KANJIVG](https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz)
- [original links](https://github.com/Gnurou/tagainijisho/blob/master/src/core/kanjidic2/CMakeLists.txt)

Refer to [kanjivg2png.py](kanjivg2png.py) and [create_jsonl.py](create_jsonl.py) to see how I preprocessing the data. Better way indeed exists.

-------------

## Environment

Refer to the requirements of huggingface [diffusers](https://github.com/huggingface/diffusers/tree/main) and its [text-to-image](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image).

-------------

## Training 

`--from_scratch` will train a unet from scratch, remove it if you just want to finetune the model.

```shell
accelerate launch --mixed_precision="fp16"  train_kanji.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=300000 \
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
  --resume_from_checkpoint="latest"
```

----------

## Testing

Replace the prompt with whatever you like.

```shell
python  test_kanji.py \
  --model_path="$MODEL_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --prompt="$PROMPT" \
  --num_images=$NUM_IMAGES \
  --checkpoint=$CHECKPOINT
```
