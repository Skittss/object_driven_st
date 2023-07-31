C_PTH=REPLACE_WITH_PATH.png
CSEM_PTH=REPLACE_WITH_PATH.png
S_PTH=REPLACE_WITH_PATH.png
SSEM_PTH=REPLACE_WITH_PATH.png

python stylize.py \
  --content_img $C_PTH \
  --target_mask $CSEM_PTH \
  --style_img $S_PTH \
  --style_mask $SSEM_PTH \
  --output_dir "./output/512" \
  --hard_width 512 \
  --iteration 1000 \
  --log_iteration 50

python stylize.py \
  --init_img "./output/512/result_final.png" \
  --content_img $C_PTH \
  --target_mask $CSEM_PTH \
  --style_img $S_PTH \
  --style_mask $SSEM_PTH \
  --output_dir "./output/600" \
  --hard_width 600 \
  --iteration 1000 \
  --log_iteration 50