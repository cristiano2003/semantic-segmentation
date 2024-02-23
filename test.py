import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

cur_path = 'semantic-segmentation'
for root, dirs, files in os.walk("src/neopolyp"):
    for f in files:
        print(f)
        print('--------')
        