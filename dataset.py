# Dateset関数の実装

import os
import glob
import torch
import pathlib
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as transforms
from torchvision.ops.boxes import box_iou

def logit(x):
    return torch.log((x + 1e-7) / (1 - x + 1e-7))

def get_image_paths_and_bbox_list(IMG_DIR, CSV_DIR):
    df = pd.read_csv(CSV_DIR)
    img_paths = []
    bbox_list = []

    img_list = []

    cls_list = []

    for index, row in df.iterrows():
        img_id = row[0]

        if "r" in row[0]:
            cls_list.append(0)
        elif "s" in row[0]:
            cls_list.append(1)
        else:
            cls_list.append(2)

        bbox = row[1]
        bbox = bbox.replace('[', '')
        bbox = bbox.replace(']', '')
        bbox = bbox.split(',')
        xmin = float(bbox[0])
        ymin = float(bbox[1])
        width = float(bbox[2])
        height = float(bbox[3])

        if img_id in img_list:
            img_index = img_list.index(img_id)

        else:
            img_list.append(img_id)
            img_path = IMG_DIR + img_id + '.jpg'
            img_paths.append(img_path)

            bbox_list.append([])
            img_index = len(bbox_list) - 1

        bbox_list[img_index].append([xmin, ymin, width, height])

    return img_paths, bbox_list, cls_list

class YOLOv3_Dataset(Dataset):
    def __init__(self, IMG_DIR, CSV_DIR, img_size_x, img_size_y, anchor_dict, trainsform=None):
        self.image_paths, self.bbox_list, self.cls_list = get_image_paths_and_bbox_list(IMG_DIR, CSV_DIR)
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.anchor_dict = anchor_dict
        self.transform = transform

        # iouの計算に使用する9行4列のテンソルを作成
        # 0、1行目には矩形領域の左上のx座標とy座標（すなわち0）
        # 2、3行目には矩形領域の右下の座標（すなわちkmeans法で求めた9通りのbboxの高さと幅）が入る
        self.anchor_iou = torch.cat([torch.zeros(9,2) , torch.tensor(self.anchor_dict[["width","height"]].values)] ,dim = 1)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path)
        if transform is not None:
            image = self.transform(image)

        scale3_targets, scale2_targets, scale1_targets = self.get_targets(self.bbox_list[index], self.cls_list[index])

        return image, scale3_targets, scale2_targets, scale1_targets

    def __len__(self):
        return len(self.image_paths)

    def get_CxCytxty(self, bbox, anchor_idx):
        '''
        inputs
        -------
        bbox : バウンティボックスの情報[x_min, y_min, width, height]
        anchor_idx : iouでもとめた、合致するアンカーの添え字

        output
        -------
        CxCytxty : [bboxの存在するグリッドのxの位置, yの位置, tx, ty]
        '''
        grid_size_list = [32, 16, 8]  # グリッドサイズ
        grid_size = grid_size_list[int(anchor_idx/3)]

        # bboxの中心座標/グリッドサイズ
        bx = (bbox[0] + (bbox[2] // 2)) / grid_size
        by = (bbox[1] + (bbox[3] // 2)) / grid_size

        # bboxの存在するグリッドの右上の座標
        Cx = int(bx)
        Cy = int(by)

        tx = logit(torch.tensor(bx - Cx))
        ty = logit(torch.tensor(by - Cy))
        CxCytxty = [Cx, Cy, tx, ty]

        return CxCytxty

    def get_twth(self, wh, anchor_idx):
        anchor = self.anchor_dict.iloc[anchor_idx]
        Pw = anchor["width"]
        Ph = anchor["height"]
        twth = [torch.log(torch.tensor(wh[0]/Pw)), torch.log(torch.tensor(wh[1]/Ph))]

        return twth

    def get_targets(self, bbox_list, cls):

        # scale3、scale2、scale1それぞれ3つずつの0で満たされたテンソルを生成
        map_size_x = [int(self.img_size_x/32) , int(self.img_size_x/16) , int(self.img_size_x/8)]  # マップサイズ
        map_size_y = [int(self.img_size_y/32) , int(self.img_size_y/16) , int(self.img_size_y/8)]  # マップサイズ
        tensor_list = []
        for size_index in range(3):
            for _ in range(3):
                tensor_list.append(torch.zeros((4 + 1 + 3 ,map_size_y[size_index],map_size_x[size_index])))

        for bbox in bbox_list:

            # iouの計算
            # kmeansでもとめたbboxとのiouを総当たりで調べて最も値が高いbboxの添え字を取得
            label_iou = torch.cat([torch.zeros((1,2))  , torch.tensor(bbox[2:]).unsqueeze(0)],dim=1)
            iou = box_iou(label_iou, self.anchor_iou).squeeze()
            obj_idx = torch.argmax(iou).item()

            for i in range(9):
                if i == obj_idx:
                    CxCytxty = self.get_CxCytxty(bbox, i)
                    twth = self.get_twth(bbox[2:], i)

                    tensor_list[i][0,CxCytxty[1],CxCytxty[0]] = CxCytxty[2]  # tx
                    tensor_list[i][1,CxCytxty[1],CxCytxty[0]] = CxCytxty[3]  # ty
                    tensor_list[i][2,CxCytxty[1],CxCytxty[0]] = twth[0]  # tw
                    tensor_list[i][3,CxCytxty[1],CxCytxty[0]] = twth[1]  #th
                    tensor_list[i][4,CxCytxty[1],CxCytxty[0]] = 1  # 確信度
                    tensor_list[i][5 + cls,CxCytxty[1],CxCytxty[0]] = 1  # スコア

        scale3_targets = torch.cat(tensor_list[0:3] , dim = 0)
        scale2_targets = torch.cat(tensor_list[3:6] , dim = 0)
        scale1_targets = torch.cat(tensor_list[6:] , dim = 0)

        return scale3_targets, scale2_targets, scale1_targets