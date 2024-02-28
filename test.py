from src.neopolyp.dataset.coco_utils import *
import matplotlib.pyplot as plt
import numpy as np

s = set()
train, val = infer_build()
for i in range(200):
    
    sample_1, sample_2 = train[i], val[i]
    image1, mask1, image2, mask2 = torch.permute(sample_1[0], (1, 2, 0)), sample_1[1], torch.permute(sample_2[0], (1, 2, 0)), sample_2[1]


plt.subplot(2, 2, 1)
plt.imshow(np.array(image1))  # You can choose a different colormap
plt.subplot(2, 2, 2)
plt.imshow(np.array(mask1))  # You can choose a different colormap
plt.subplot(2, 2, 3)
plt.imshow(np.array(image2), interpolation='nearest')  # You can choose a different colormap
plt.subplot(2, 2, 4)
plt.imshow(np.array(mask2)) 

plt.show()
