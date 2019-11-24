import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sixd import SixdToolkit


bench = SixdToolkit(dataset='hinterstoisser', unit=1e-3, num_kp=17, type_kp='sift', is_train=True)


img = Image.open('/home/yusheng/code/render/17/sift/images/00000.png')
annots = np.load('/home/yusheng/code/render/17/sift/annots/00000.npy').item()

# print(annots)
bboxes = annots['bboxes']
kps = annots['kps']
poses = annots['poses']
obj_ids = annots['obj_ids']

bboxes[:,2] -= bboxes[:,0]
bboxes[:,3] -= bboxes[:,1]

fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')

for IDX in range(bboxes.shape[0]):
    models = bench.models['%02d' % annots['obj_ids'][IDX]]
    corners = bench.get_3d_corners(models)

    corners = np.concatenate((corners, np.ones((corners.shape[0], 1))), axis=1)
    projected = np.matmul(np.matmul(bench.cam, poses[IDX]), corners.T)
    projected /= projected[2, :]
    projected = projected[:2, :].T

    edges_corners = (
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    )
    ax.scatter(projected[:, 0], projected[:, 1], s=10, c='g')
    for edge in edges_corners:
        ax.plot(projected[edge, 0], projected[edge, 1], linewidth=1.0, c='g')

    bbox = bboxes[IDX]
    rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    kp = kps[IDX]
    ax.scatter(kp[:, 0], kp[:, 1], c='aqua', marker='x', s=3)


plt.savefig('demo.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
