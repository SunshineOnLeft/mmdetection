import time
import os
import json
import argparse
import json


def xywh2xyxy(x, y, w, h):
    return x, y, x + w, y + h


def shifted_xyxy(x1, y1, x2, y2, w_id, h_id):
    l = x1 + w_id
    t = y1 + h_id
    r = x2 + w_id
    b = y2 + h_id
    return l, t, r, b


def main(bbox_json, image_json):
    f = open(bbox_json)
    content = f.read()
    f.close
    bbox_dict = json.loads(content)

    f = open(image_json)
    content = f.read()
    f.close
    image_dict = json.loads(content)   

    save_path = './result/' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json_name = os.path.join(save_path, '%s_result.json' %time.strftime("%Y%m%d%H%M%S", time.localtime()))

    result = []
    bbox_num = len(bbox_dict)
    for logs_id, logs in enumerate(bbox_dict, 1):
        print("%s/%s" %(logs_id, bbox_num))

        img_log = image_dict['images'][logs['image_id']]
        assert img_log['id'] == logs['image_id']
        img_name = img_log['file_name'] 
        img_name_main = img_name[:-4]
        h_id = float(img_name_main.split("_")[-2])
        w_id = float(img_name_main.split("_")[-1])
        img_name_ori = '_'.join(img_name_main.split("_")[:-2]) + img_name[-4:]

        x, y, w, h = logs['bbox']
        x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)
        x1, y1, x2, y2 = shifted_xyxy(x1, y1, x2, y2, w_id, h_id)
        x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  #save 0.00
        result.append({'name': img_name_ori, 
                       'category': logs['category_id'], 
                       'bbox': [x1, y1, x2, y2], 
                       'score': logs['score']})

    with open(json_name, 'w') as fp:
        # json.dump(result, fp, separators=(',', ':'), ensure_ascii=False, indent=4)
        json.dump(result, fp, separators=(',', ':'), ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bbox_json', required=True, type=str, help='json file of inference result')
    parser.add_argument('--image_json', required=True, type=str, help='json file as annotations of testing images')
    opt = parser.parse_args()

    main(opt.bbox_json, opt.image_json)