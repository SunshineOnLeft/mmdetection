import cv2
import torch
import os
import numpy as np
import json


ANNOS = '/data/datasets/det/tile_surface_defect_detection/tile_round1_train_20201231/train_annos.json'
RPN_LOSS = 0
RPN_GET_BBOX = 0
ROI_LOSS = 0
ROI_TEST_PROPOSAL = 0
ROI_TEST_GET_BBOX_AFTER_NMS = 0


def rpn_loss_save_images(
        img_metas, # (batch, [...])
        cls_scores, # (pyramid, [batch, num_anchors * num_classes, hi, wi]), num_classes = 1 for RPN
        bbox_preds, # (pyramid, [batch, num_anchors * 4, hi, wi])
        anchor_list, # (pyramid, [batch, num_anchors * hi * wi, 4])
        labels_list, # (pyramid, [batch, num_anchors * hi * wi])
        gt_bboxes, # (batch, [num_gt, 4])
        bbox_coder):

    if RPN_LOSS:
        for j in range(len(img_metas)):
            # flag = False
            '''
            fn = img_metas[j]['filename']
            if len(self.image_set) >= self.image_num:
                if fn in self.image_set:
                    flag = True
            else:
                self.image_set[fn] = True
                flag = True
            '''
            # if img_metas[j]['filename'][-12:-4] in cfg.save_list:
            #     flag = True

            # if flag:
            pyramid = len(cls_scores)
            num_class = cls_scores[0].size(1) // 3

            labels_list_show = torch.cat([labels_list[i][j] for i in range(pyramid)])
            bbox_preds_show = torch.cat([bbox_preds[i][j].permute(1, 2, 0).reshape(-1, 4) for i in range(pyramid)])
            anchor_list_show = torch.cat([anchor_list[i][j] for i in range(pyramid)])
            cls_scores_show = torch.cat([cls_scores[i][j].permute(1, 2, 0).reshape(-1, num_class).sigmoid() for i in range(pyramid)])

            _, topk_inds = cls_scores_show[labels_list_show == 0].topk(len((labels_list_show == 0).nonzero()))
            topk_inds = (labels_list_show == 0).nonzero().reshape(-1)[topk_inds]
            bbox_preds_show = bbox_coder.decode(anchor_list_show, bbox_preds_show)
                                            
            labels_list_show = labels_list_show[topk_inds]
            bbox_preds_show = bbox_preds_show[topk_inds]
            cls_scores_show = cls_scores_show[topk_inds]

            img = cv2.imread(img_metas[j]['filename'])

            img = cv2.resize(img, (img_metas[j]['pad_shape'][1], img_metas[j]['pad_shape'][0]))
            if img_metas[j]['flip']:
                cv2.flip(img, 1, img)
            for i, bbox in enumerate(bbox_preds_show):
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, 'score:%.3f rank:%d label:%d' % (cls_scores_show[i].item(), i+1, labels_list_show[i].item()), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for gt_bbox in gt_bboxes[j]:
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), 1)
            # for point in points:
                # cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
                #cv2.circle(img, center, radius, color)
            save_path = './images/rpn_loss/'
            save_name = os.path.join(save_path, '%s.png' %(img_metas[j]['filename'][-12:-4]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            cv2.imwrite(save_name, img)
            print("save %s" %save_name)


def rpn_get_bbox_save_images(
        img_metas, # (batch, [...])
        result_list, # (batch, [2000, 5])
        gt_bboxes): # (batch, [num_gt, 4])

    if RPN_GET_BBOX:
        for j in range(len(img_metas)):
            topk = 50
            bbox_preds_show = result_list[j][:topk, :-1]
            cls_scores_show = result_list[j][:topk, -1]
            labels_list_show = torch.zeros_like(cls_scores_show)

            img = cv2.imread(img_metas[j]['filename'])

            img = cv2.resize(img, (img_metas[j]['pad_shape'][1], img_metas[j]['pad_shape'][0]))
            if img_metas[j]['flip']:
                cv2.flip(img, 1, img)
            for i, bbox in enumerate(bbox_preds_show):
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, 'score:%.3f rank:%d label:%d' % (cls_scores_show[i].item(), i+1, labels_list_show[i].item()), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for gt_bbox in gt_bboxes[j]:
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), 1)
            # for point in points:
                # cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
                #cv2.circle(img, center, radius, color)
            save_path = './images/rpn_get_bbox/'
            save_name = os.path.join(save_path, '%s.png' %(img_metas[j]['filename'][-12:-4]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            cv2.imwrite(save_name, img)
            print("save %s" %save_name)


def roi_loss_save_images(
        img_metas, # (batch, [...])
        cls_scores, # [num, num_class + 1]
        bbox_preds, # [num, 4]
        anchor_list, # [num, 5]
        labels_list, # [num]
        gt_bboxes, # (batch, [num_gt, 4])
        bbox_coder,
        stage):

    if ROI_LOSS and stage == 2:
        for j in range(len(img_metas)):
            inds = (anchor_list[:, 0] == j).nonzero().reshape(-1)
            cls_scores_show = cls_scores[inds]
            bbox_preds_show = bbox_preds[inds]
            anchor_list_show = anchor_list[inds, 1:]
            labels_list_show = labels_list[inds]

            num_class = cls_scores.size(1) - 1

            cls_scores_show = cls_scores_show[range(cls_scores_show.size(0)), labels_list_show]

            _, topk_inds = cls_scores_show[labels_list_show != num_class].topk(len((labels_list_show != num_class).nonzero()))
            topk_inds = (labels_list_show != num_class).nonzero().reshape(-1)[topk_inds]
            bbox_preds_show = bbox_coder.decode(anchor_list_show, bbox_preds_show)
                                            
            labels_list_show = labels_list_show[topk_inds]
            bbox_preds_show = bbox_preds_show[topk_inds]
            cls_scores_show = cls_scores_show[topk_inds]

            img = cv2.imread(img_metas[j]['filename'])

            img = cv2.resize(img, (img_metas[j]['pad_shape'][1], img_metas[j]['pad_shape'][0]))
            if img_metas[j]['flip']:
                cv2.flip(img, 1, img)
            for i, bbox in enumerate(bbox_preds_show):
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, 'score:%.3f rank:%d label:%d' % (cls_scores_show[i].item(), i+1, labels_list_show[i].item()), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for gt_bbox in gt_bboxes[j]:
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), 1)
            # for point in points:
                # cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
                #cv2.circle(img, center, radius, color)
            save_path = './images/roi_loss/'
            save_name = os.path.join(save_path, 'stage%d_%s.png' %(stage, img_metas[j]['filename'][-12:-4]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            cv2.imwrite(save_name, img)
            print("save %s" %save_name)


def roi_test_proposal_save_images(
        img_metas,  # (batch=1, [...])
        scores, # [num, num_class + 1]
        bboxes, # [num, 4]
        gt_bboxes=None):

    if ROI_TEST_PROPOSAL:
        f = open(ANNOS)
        content = f.read()
        f.close
        tianchi_json = json.loads(content)

        img_metas = img_metas[0]
        cls_scores_show, labels_list_show = scores.max(1)
        bbox_preds_show = bboxes

        num_class = scores.size(1) - 1

        _, topk_inds = cls_scores_show[labels_list_show != num_class].topk(len((labels_list_show != num_class).nonzero()))
        topk_inds = (labels_list_show != num_class).nonzero().reshape(-1)[topk_inds]
                                        
        labels_list_show = labels_list_show[topk_inds]
        bbox_preds_show = bbox_preds_show[topk_inds].int()
        cls_scores_show = cls_scores_show[topk_inds]

        img = cv2.imread(img_metas['filename'])

        if img_metas['flip']:
            cv2.flip(img, 1, img)
        for i, bbox in enumerate(bbox_preds_show):
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=4)
            cv2.putText(img, 'score:%.3f rank:%d label:%d' % (cls_scores_show[i].item(), i+1, labels_list_show[i].item()), (bbox[0].item(), bbox[1].item()), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 1)
        if gt_bboxes is None:
            gt_bboxes = []
            for logs in tianchi_json:
                if logs['name'] == img_metas['filename'].split('/')[-1]:
                    gt_bboxes.append(logs['bbox'])
        for gt_bbox in gt_bboxes:
            gt_bbox = torch.tensor(gt_bbox).int()
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), thickness=4)
        # for point in points:
            # cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
            #cv2.circle(img, center, radius, color)
        save_path = './images/roi_test_proposal/'
        save_name = os.path.join(save_path, '%s.png' %(img_metas['filename'][-12:-4]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img = cv2.resize(img, (img_metas['pad_shape'][1], img_metas['pad_shape'][0]))
        cv2.imwrite(save_name, img)
        print("save %s" %save_name)


def roi_test_get_bbox_after_nms_save_images(
        img_metas, # (batch=1, [...])
        det_bboxes, # (batch=1, [100, 5])
        det_labels, # (batch=1, [100])
        gt_bboxes=None):

    if ROI_TEST_GET_BBOX_AFTER_NMS:
        f = open(ANNOS)
        content = f.read()
        f.close
        tianchi_json = json.loads(content)
        
        for j in range(len(img_metas)):
            topk = len(det_labels[0])
            # topk = 2
            bbox_preds_show = det_bboxes[j][:topk, :-1].int()
            cls_scores_show = det_bboxes[j][:topk, -1]
            labels_list_show = det_labels[j][:topk]

            img = cv2.imread(img_metas[j]['filename'])

            if img_metas[j]['flip']:
                cv2.flip(img, 1, img)
            for i, bbox in enumerate(bbox_preds_show):
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=4)
                cv2.putText(img, 'score:%.3f rank:%d label:%d' % (cls_scores_show[i].item(), i+1, labels_list_show[i].item()), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 1)
            if gt_bboxes is None:
                gt_bboxes = []
                for logs in tianchi_json:
                    if logs['name'] == img_metas[j]['filename'].split('/')[-1]:
                        gt_bboxes.append(logs['bbox'])
            for gt_bbox in gt_bboxes:
                gt_bbox = torch.tensor(gt_bbox).int()
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), thickness=4)
            # for point in points:
                # cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
                #cv2.circle(img, center, radius, color)
            save_path = './images/roi_test_get_bbox_after_nms/'
            save_name = os.path.join(save_path, '%s.png' %(img_metas[j]['filename'][-12:-4]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            img = cv2.resize(img, (img_metas[j]['pad_shape'][1], img_metas[j]['pad_shape'][0]))
            cv2.imwrite(save_name, img)
            print("save %s" %save_name)