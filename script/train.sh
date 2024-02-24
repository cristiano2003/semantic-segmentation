python -m neopolyp.train \
        --model unet \
        --batch_size 32 \
        --max_epochs 25 \
        --num_workers 4 \
        --lr 0.0001 \
        --split_ratio 0.9 \
        --data_path pascal-voc-2012-dataset
        # -w -wk 53f5746150b2ce7b0552996cb6acc3beec6e487f