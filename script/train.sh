python -m neopolyp.train \
        --model unet \
        --batch_size 64 \
        --max_epochs 50 \
        --num_workers 4 \
        --lr 0.0001 \
        --split_ratio 0.9 \
        --data_path bkai-igh-neopolyp 
        