import os

all_path = []
image_path = 'dataset/training/image_2'
for file in os.listdir(image_path):
    all_path.append(os.path.join(image_path, file))
    
all_gt_path = []
image_gt_path = 'dataset/training/gt_image_2'
for file in os.listdir(image_gt_path):
    all_gt_path.append(os.path.join(image_path, file))
        
print(all_path[0])
print(all_gt_path[0])    