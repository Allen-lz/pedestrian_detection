import os
import sys
import argparse

import cv2
import torch
import numpy as np

sys.path.insert(0, '../lib')
from utils import visual_utils, nms_utils

def inference(config, network):
    # model_path
    saveDir = os.path.join('../model', 'rcnn_emd_simple')
    model_file = os.path.join(saveDir, 'outputs', 'rcnn_emd_simple_mge.pth')
    assert os.path.exists(model_file)
    # build network
    net = network()
    net.eval()
    check_point = torch.load(model_file, map_location=torch.device('cpu'))
    net.load_state_dict(check_point['state_dict'])
    # get data
    video = "../test_video/Crash_walk_f_cm_np1_ri_med_0.avi"
    cap = cv2.VideoCapture(video)
    retaining = True
    # 视频帧计数间隔频率
    timeF = 4
    # 记录积累的帧数
    c = 0
    while retaining:
        retaining, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not retaining and frame is None:
            continue
        if c % timeF == 0:
            image, resized_img, im_info = get_numpy_data(
                    frame, config.eval_image_short_size, config.eval_image_max_size)
            pred_boxes = net(resized_img, im_info).numpy()
            pred_boxes = post_process(pred_boxes, config, im_info[0, 2])

            persons = visual_utils.draw_boxes(
                    image,
                    pred_boxes[:, :4],)
            # 保存一张从一张图中检测出来的行人
            for person_id, person in enumerate(persons):
                fpath = 'outputs/{}_{}.png'.format(str(c), 'person' + '_' + str(person_id + 1))
                print(fpath)
                cv2.imwrite(fpath, person)
        c += 1

def post_process(pred_boxes, config, scale):
    if config.test_nms_method == 'set_nms':
        assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        top_k = pred_boxes.shape[-1] // 6
        n = pred_boxes.shape[0]
        pred_boxes = pred_boxes.reshape(-1, 6)
        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'normal_nms':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'none':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]

    pred_boxes[:, :4] /= scale
    keep = pred_boxes[:, 4] > config.visulize_threshold
    pred_boxes = pred_boxes[keep]
    return pred_boxes

def get_data(img_path, short_size, max_size):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized_img, scale = resize_img(
            image, short_size, max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    resized_img = resized_img.transpose(2, 0, 1)
    im_info = np.array([height, width, scale, original_height, original_width, 0])
    return image, torch.tensor([resized_img]).float(), torch.tensor([im_info])

def get_numpy_data(image, short_size, max_size):
    resized_img, scale = resize_img(
            image, short_size, max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    resized_img = resized_img.transpose(2, 0, 1)
    im_info = np.array([height, width, scale, original_height, original_width, 0])
    return image, torch.tensor([resized_img]).float(), torch.tensor([im_info])

def resize_img(image, short_size, max_size):
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    resized_image = cv2.resize(
            image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale

def run_inference():
    # import libs
    model_root_dir = os.path.join('../model/', 'rcnn_emd_simple')
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    inference(config, Network)

if __name__ == '__main__':
    run_inference()
