# アンカーボックスを定める
import os
import glob
import pandas as pd
from sklearn.cluster import KMeans
import torch
import pathlib
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def get_anchor_dict(bbox_path, img_size):
    df = pd.read_csv(bbox_path)

    bbox_dict = {'width':[] , 'height':[]}
    bbox_list = []

    for index, row in df.iterrows():
        bbox = row[1]
        bbox = bbox.replace('[', '')
        bbox = bbox.replace(']', '')
        bbox = bbox.split(',')

        width = float(bbox[2])
        height = float(bbox[3])

        bbox_dict['width'].append(width)
        bbox_dict['height'].append(height)

        bbox_list.append([width, height])

    bbox_df = pd.DataFrame(bbox_dict)

    km = KMeans(n_clusters=9,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    km_cluster = km.fit_predict(bbox_list)

    bbox_df['cluster'] = km_cluster
    anchor_wha = {"width":[],"height":[],"area":[]}
    for i in range(9):
        anchor_wha["width"].append(bbox_df[bbox_df["cluster"] == i].mean()["width"])
        anchor_wha["height"].append(bbox_df[bbox_df["cluster"] == i].mean()["height"])
        anchor_wha["area"].append(bbox_df[bbox_df["cluster"] == i].mean()["width"]*bbox_df[bbox_df["cluster"] == i].mean()["height"])

    anchor_dict = pd.DataFrame(anchor_wha).sort_values('area', ascending=False)
    anchor_dict["type"] = [int(img_size/32) ,int(img_size/32) ,int(img_size/32) ,  int(img_size/16) ,int(img_size/16) ,int(img_size/16) , int(img_size/8), int(img_size/8), int(img_size/8)]

    return anchor_dict

def get_bbox(tensor, x, y, grid_size, anchor):
    tx = tensor[0, y, x]
    ty = tensor[1, y, x]
    tw = tensor[2, y, x]
    th = tensor[3, y, x]

    bx = (torch.sigmoid(tx) + x) * grid_size
    by = (torch.sigmoid(ty) + y) * grid_size

    Pw = anchor["width"]
    Ph = anchor["height"]
    bw = Pw * torch.exp(tw)
    bh = Ph * torch.exp(th)

    xmin = int(bx - (bw / 2))
    if xmin < 0:
        xmin = 0
    ymin = int(by - (bh / 2))
    if ymin < 0:
        ymin = 0
    xmax = int(bx + (bw / 2))
    if xmax > 640:
        xmax = 640
    ymax = int(by + (bh / 2))
    if ymax > 480:
        ymax = 480

    return [xmin, ymin, xmax, ymax]

def get_multiple_bboxes(pred, anchor_dict, num_cls, img_size_x, img_size_y, conf_threshold=0.01):
    bbox_list = []
    conf_list = []
    label_list = []
    score_list = []
    NMS_bbox_list = []
    NMS_conf_list = []
    NMS_label_list = []
    NMS_score_list = []
    flag = False

    anchor_idx = 0
    grid_size_list = [32, 16, 8]  # グリッドサイズ
    for i, scale in enumerate(pred):
        grid_size = grid_size_list[i]
        tensor1 = scale[0:8]
        tensor2 = scale[8:16]
        tensor3 = scale[16:24]
        tensor_list = [tensor1, tensor2, tensor3]
        for tensor in tensor_list:
            conf = tensor[4, :, :]
            grid_x = int(img_size_x / grid_size)
            grid_y = int(img_size_y / grid_size)
            for y in range(grid_y):
                for x in range(grid_x):
                    if conf[y, x] >= conf_threshold:  # 確信度が閾値を超えた場合
                        bbox = get_bbox(tensor, x, y, grid_size, anchor_dict.iloc[anchor_idx])
                        bbox_list.append(bbox)
                        conf_list.append(conf[y, x])
                        score = 0
                        label = None
                        for cls_index in range(num_cls):
                            if score < torch.sigmoid(tensor[5 + cls_index, y, x]):
                                label = cls_index
                                score = torch.sigmoid(tensor[5 + cls_index, y, x])
                        label_list.append(label)
                        score_list.append(score)

            anchor_idx += 1

    bbox_nms_inputs = torch.tensor(bbox_list, dtype=torch.float32)
    conf_nms_inputs = torch.tensor(conf_list, dtype=torch.float32)

    # bboxを検出できなかった場合
    if bbox_nms_inputs.size(0) == 0:
        return flag, bbox_list, label_list, score_list, NMS_bbox_list, NMS_conf_list, NMS_label_list, NMS_score_list

    # 1つ以上のbboxを検出できた場合
    else:
        flag = True
        nms_idx_list = nms(bbox_nms_inputs, conf_nms_inputs, iou_threshold=0.5)

        for nms_idx in nms_idx_list:
            NMS_bbox_list.append(bbox_list[nms_idx])
            NMS_conf_list.append(conf_list[nms_idx])
            NMS_label_list.append(label_list[nms_idx])
            NMS_score_list.append(score_list[nms_idx])

        bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
        label_list = torch.tensor(label_list, dtype=torch.float32)
        score_list = torch.tensor(score_list, dtype=torch.float32)
        NMS_bbox_list = torch.tensor(NMS_bbox_list, dtype=torch.float32)
        NMS_conf_list = torch.tensor(NMS_conf_list, dtype=torch.float32)
        NMS_label_list = torch.tensor(NMS_label_list, dtype=torch.float32)
        NMS_score_list = torch.tensor(NMS_score_list, dtype=torch.float32)

        #print(bbox_list.size())
        #print(NMS_bbox_list.size())

        return flag, bbox_list, label_list, score_list, NMS_bbox_list, NMS_conf_list, NMS_label_list, NMS_score_list

def draw_bbox(img, bbox_list, cls_list=None, labels=None):
    # 前処理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.PILToTensor()
    ])
    pil_image = transform(img)

    colors = (255, 0, 0)

    # 線の太さを指定
    line_width = 3

    if cls_list != None:
        label_list = []
        for cls in cls_list:
            label_list.append(labels[int(cls)])

        drawn_image = draw_bounding_boxes(pil_image, boxes=bbox_list, labels=label_list, colors=colors, width=line_width)
        print(label_list)
    else:
        # bboxの描画
        drawn_image = draw_bounding_boxes(pil_image, bbox_list, colors=colors, width=line_width)

    # テンソル画像をPIL画像に変換
    pil_image = TF.to_pil_image(drawn_image)

    # PIL画像を表示する
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()

# 損失関数の実装
def YOLOv3_loss_function(pred_list, targets_list, lambda_coord=0.1, lambda_obj=1.0, lambda_class=1.0, lambda_noobj=0.01):

    coord_loss = 0
    obj_loss = 0
    class_loss = 0
    noobj_loss = 0

    B = pred_list[0].size(0)

    for i in range(3):
        pred = pred_list[i]
        targets = targets_list[i]
        for j in range(3):
            pred_cut = pred[:, j*(4+1+3):(j+1)*(4+1+3), :, :]  # predの分割
            pred_boxes = pred_cut[:, :4, :, :]  # 予測されたbboxの座標 (tx, ty, tw, th)
            pred_conf_obj = pred_cut[:, 4, :, :].unsqueeze(dim=1)  # 予測されたbboxの確信度
            pred_classes = pred_cut[:, 5:, :, :]  # クラスのスコア

            targets_cut = targets[:, j*(4+1+3):(j+1)*(4+1+3), :, :]  # targetsの分割
            targets_boxes = targets_cut[:, :4, :, :]  # targetsのbboxの座標 (tx, ty, tw, th)
            targets_obj_mask = targets_cut[:, 4, :, :].unsqueeze(dim=1)  # オブジェクトマスク
            targets_classes = targets_cut[:, 5:, :, :]  # targetsのスコア

            # 位置の損失を計算
            #coord_loss += lambda_coord * nn.MSELoss(reduction='sum')(targets_obj_mask * pred_boxes, targets_boxes)
            coord_loss += lambda_coord * torch.sum(torch.square(pred_boxes - targets_boxes) * targets_obj_mask)

            # オブジェクトの損失を計算
            #obj_loss += lambda_obj * nn.BCEWithLogitsLoss(reduction='sum')(targets_obj_mask * pred_conf_obj, targets_obj_mask)
            obj_loss += lambda_obj * torch.sum(-1 * torch.log(torch.sigmoid(pred_conf_obj)+ 1e-7) * targets_obj_mask)

            # クラスの損失を計算
            class_loss += lambda_class * torch.sum(targets_obj_mask * F.binary_cross_entropy(torch.sigmoid(pred_classes), targets_classes))

            # ノンオブジェクトの損失を計算
            #noobj_loss += lambda_noobj * nn.BCEWithLogitsLoss(reduction='sum')((1 - targets_obj_mask) * (1 - pred_conf_obj), 1 - targets_obj_mask)
            noobj_loss += lambda_noobj * torch.sum((-1 * torch.log(1 - torch.sigmoid(pred_conf_obj)+ 1e-7)) * (1 - targets_obj_mask))

    loss = coord_loss + obj_loss + class_loss + noobj_loss

    return loss/B, coord_loss/B, obj_loss/B, class_loss/B, noobj_loss/B