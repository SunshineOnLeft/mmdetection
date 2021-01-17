import time
import os
import json
import mmcv 
from mmdet.apis import init_detector, inference_detector
import argparse


too_small = 0


def _shrink_xyxy(bbox, shrink=16.):
    global too_small
    _bbox = bbox.copy()
    l = _bbox[0] + shrink
    t = _bbox[1] + shrink
    r = _bbox[2] - shrink
    b = _bbox[3] - shrink
    if l >= r:
        r = l + 1.
        too_small += 1
        print(too_small)
    if t >= b:
        b = t + 1
        too_small += 1
        print(too_small)        

    assert l < r and t < b
    return [l, t, r, b]


def main(config_name, shrink_coefficient):
    config_file = 'local_configs/tianchi/%s.py' %config_name # 修改成自己的配置文件
    checkpoint_file = 'work_dirs/%s/latest.pth' %config_name # 修改成自己的训练权重
    test_path = '/data/datasets/det/tile_surface_defect_detection/tile_round1_testA_20201231/cut640_imgs/' # 官方测试集图片路径
    save_path = 'result/' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json_name = os.path.join(save_path, '%s_result.json' %time.strftime("%Y%m%d%H%M%S", time.localtime()))
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    image_num = len(img_list)
    for img_id, img_name in enumerate(img_list, 1):
        print("%s / %s" %(img_id, image_num))
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes)>0:
                defect_label = i
                # print(i)
                image_name = img_name
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  #save 0.00
                    result.append({'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

    for logs in result:
        bbox = logs['bbox']
        logs['bbox'] = _shrink_xyxy(bbox, shrink_coefficient)

    with open(json_name, 'w') as fp:
        json.dump(result, fp, separators=(',', ':'), ensure_ascii=False, indent=4)


def shrink():
    shrink_coefficient = 16.

    f = open('result/result.json')
    content = f.read()
    f.close
    ori_json = json.loads(content)

    save_path = 'result/' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json_name = os.path.join(save_path, '%s_result.json' %time.strftime("%Y%m%d%H%M%S", time.localtime()))

    for logs in ori_json:
        bbox = logs['bbox']
        logs['bbox'] = _shrink_xyxy(bbox, shrink_coefficient)

    with open(json_name, 'w') as fp:
        json.dump(ori_json, fp, separators=(',', ':'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', required=True, type=str, help='config name')
    parser.add_argument('--shrink_coefficient', required=True, type=float, help='shrink coefficient')
    opt = parser.parse_args()

    main(opt.config_name, opt.shrink_coefficient)
    # shrink()