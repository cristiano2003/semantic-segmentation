all_path = []
import os
data_path = 'TrainDataset/image'
for file in os.listdir(data_path):
    all_path.append(os.path.join(data_path, file))
