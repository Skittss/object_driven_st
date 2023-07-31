C_PTH=src/arcimboldo_exemplars/four_seasons/ref.png
CSEM_PTH=src/arcimboldo_exemplars/four_seasons/ref_sem.png
S_PTH=src/arcimboldo_exemplars/four_seasons/style.png
SSEM_PTH=src/arcimboldo_exemplars/four_seasons/style_sem.png

python src/gatys2017/spatial_control.py \
  --content $C_PTH --content-sem $CSEM_PTH \
  --style $S_PTH --style-sem $SSEM_PTH \
  -d "./exports/32" \
  --image-width 32 \
  --num-iter 1000 \
  --interval 50 \
  --fixed-iter

python src/gatys2017/spatial_control.py \
  --content $C_PTH --content-sem $CSEM_PTH \
  --style $S_PTH --style-sem $SSEM_PTH \
  --initial-img "./exports/32/best_result.png" \
  -d "./exports/64" \
  --image-width 64 \
  --num-iter 1000 \
  --interval 50 \
  --fixed-iter

python src/gatys2017/spatial_control.py \
  --content $C_PTH --content-sem $CSEM_PTH \
  --style $S_PTH --style-sem $SSEM_PTH \
  --initial-img "./exports/64/best_result.png" \
  --d "./exports/128" \
  --image-width 128 \
  --num-iter 1000 \
  --interval 50 \
  --fixed-iter

python src/gatys2017/spatial_control.py \
  --content $C_PTH --content-sem $CSEM_PTH \
  --style $S_PTH --style-sem $SSEM_PTH \
  --initial-img "./exports/128/best_result.png" \
  -d "./exports/256" \
  --image-width 256 \
  --num-iter 1000 \
  --interval 50 \
  --fixed-iter

python src/gatys2017/spatial_control.py \
  --content $C_PTH --content-sem $CSEM_PTH \
  --style $S_PTH --style-sem $SSEM_PTH \
  --initial-img "./exports/256/best_result.png" \
  -d "./exports/512" \
  --image-width 512 \
  --num-iter 1000 \
  --interval 50 \
  --fixed-iter