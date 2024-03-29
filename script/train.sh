python -m segmentation.train \
        --model resunet \
        --batch_size 16 \
        --max_epochs 40 \
        --num_workers 4 \
        --lr 0.0001 \
        --data_path /kaggle/input/coco-2017-dataset/coco2017 \
        -w -wk 53f5746150b2ce7b0552996cb6acc3beec6e487f