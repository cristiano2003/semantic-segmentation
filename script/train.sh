python -m neopolyp.train \
        --model segresnet \
        --batch_size 16 \
        --max_epochs 25 \
        --num_workers 4 \
        --lr 0.0001 \
        --split_ratio 0.9 \
        --masks \
        --coco_path /kaggle/input/coco-val/data \
        -w -wk 53f5746150b2ce7b0552996cb6acc3beec6e487f