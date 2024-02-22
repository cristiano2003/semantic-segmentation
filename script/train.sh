python -m neopolyp.train \
        --model unet \
        --batch_size 16 \
        --max_epochs 50 \
        --num_workers 4 \
        --lr 0.0001 \
        --split_ratio 0.96 \
        --data_path bkai-igh-neopolyp \
        -w -wk 53f5746150b2ce7b0552996cb6acc3beec6e487f