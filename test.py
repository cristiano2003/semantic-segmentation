from src.neopolyp.dataset.coco_utils import *

train, val = infer_build()
for idx in range(2000):
    sample_1, sample_2 = train[idx], val[idx]
    image1, mask1, image2, mask2 = torch.permute(sample_1[0], (1, 2, 0)), sample_1[1], torch.permute(sample_2[0], (1, 2, 0)), sample_2[1]

    s = set()
    for i in mask2:
        for j in i:
            s.add(j.item())
            if j.item() > 91: exit()
    print(s)

print(len(s))