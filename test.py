from src.neopolyp.dataset.coco_utils import *

train, val = infer_build("val")

sample_1, sample_2 = train[0], val[0]
image1, mask1, image2, mask2 = torch.permute(sample_1[0], (1, 2, 0)), sample_1[1], torch.permute(sample_2[0], (1, 2, 0)), sample_2[1]

import matplotlib.pyplot as plt
import numpy as np


# Display the RGB image
plt.subplot(2, 2, 1)
plt.imshow(np.array(image1))
plt.title('image 1')
plt.axis('off')

# Display the mask
plt.subplot(2, 2, 2)
plt.imshow(np.array(mask1))  # You can choose a different colormap
plt.title('Mask 2')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(np.array(image2))  # You can choose a different colormap
plt.title('Mask 2')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.array(mask2) )  # You can choose a different colormap
plt.title('Mask 2')
plt.axis('off')

plt.show()
