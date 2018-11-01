import cv2
import numpy as np
from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu
from second.core.box_np_ops import riou_cc

print(rotate_iou_gpu)

def cv2_rbbx_overlaps(boxes, query_boxes):
    '''
    Parameters
    ----------------
    boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
    query: (K, 5) --- x_ctr, y_ctr, height, width, angle
    ----------------
    Returns
    ----------------
    Overlaps (N, K) IoU
    '''
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype = np.float32)

    for k in range(K):
        query_area = query_boxes[k, 2] * query_boxes[k, 3]
        for n in range(N):
            box_area = boxes[n, 2] * boxes[n, 3]
            #IoU of rotated rectangle
            #loading data anti to clock-wise
            rn = ((boxes[n, 0], boxes[n, 1]), (boxes[n, 2], boxes[n, 3]), -boxes[n, 4])
            rk = ((query_boxes[k, 0], query_boxes[k, 1]), (query_boxes[k, 2], query_boxes[k, 3]), -query_boxes[k, 4])
            int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]

            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints = True)
                int_area = cv2.contourArea(order_pts)
                overlaps[n, k] = int_area * 1.0 / (query_area + box_area - int_area)
    return overlaps

# (x,y,w,l,r)
box1 = np.array([[0, 0, 2, 2, 0.3]], dtype=np.float32)
box2 = np.array([[2, 0, 4, 1, 0.3]], dtype=np.float32)
iou1 = rotate_iou_gpu(box1, box2) # 1/7
print(iou1)

iou2 = riou_cc(box1, box2) # 1/7
print(iou2)

box1[..., [0,1]] += 100
box2[..., [0,1]] += 100
iou3 = cv2_rbbx_overlaps(box1, box2)
print(iou3)
