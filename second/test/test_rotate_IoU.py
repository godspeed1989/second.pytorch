import numpy as np
from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu

print(rotate_iou_gpu)

# (x,y,w,l,r)
box1 = np.array([[0, 0, 2, 2, 0]], dtype=np.float32)
box2 = np.array([[2, 0, 4, 1, 0]], dtype=np.float32)
iou12 = rotate_iou_gpu(box1, box2) # 1/7
print(iou12)
