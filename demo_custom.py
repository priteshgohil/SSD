import argparse
import shutil
import random
import numpy as np
import time
import torch
from PIL import Image

from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dataset import *
from ssd.data.datasets import COCODataset, VOCDataset
import ssd.utils.torch_utils as torch_utils
from vizer.draw import draw_boxes



def detect(cfg, ckpt, score_threshold, output_dir, dataset_type):
    # imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    # out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    source, half = args.source, args.half
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")
    # device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # delete output folder
    os.makedirs(output_dir)  # make new output folder

    # Initialize model
    model = build_detection_model(cfg)

    # Load weights
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz)
        print("Implement video loader.... Exiting code.")
        exit()
    else:
        save_img = True
        dataset = LoadImages(source, img_size=cfg.INPUT.IMAGE_SIZE)

    # Get names and colors
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Define class names...')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for i,(path, img, im0s, vid_cap) in enumerate(dataset): # img is resized RGB, im0s is original BGR
        image_name = os.path.basename(path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        result = model(img)[0]
        t2 = torch_utils.time_synchronized()
        inference_time = torch_utils.time_synchronized() - t1

        # to float
        if half:
            pred = pred.float()

        width, height = im0s.shape[:2]
        print("original widht height {}:{}".format(width,height))
        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(path), image_name, meters))
        print(result)
        drawn_image = draw_boxes(im0s, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
        print('%sDone. (%.3fs)' % (path, t2 - t1))
    #
    #     # Apply NMS
    #     pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
    #                                multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
    #
    #     # Apply Classifier
    #     if classify:
    #         pred = apply_classifier(pred, modelc, img, im0s)
    #
    #     # Process detections
    #     for i, det in enumerate(pred):  # detections for image i
    #         if webcam:  # batch_size >= 1
    #             p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
    #         else:
    #             p, s, im0 = path, '', im0s
    #
    #         save_path = str(Path(out) / Path(p).name)
    #         s += '%gx%g ' % img.shape[2:]  # print string
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
    #         if det is not None and len(det):
    #             # Rescale boxes from imgsz to im0 size
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class
    #                 s += '%g %ss, ' % (n, names[int(c)])  # add to string
    #
    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 if save_txt:  # Write to file
    #                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
    #                         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
    #
    #                 if save_img or view_img:  # Add bbox to image
    #                     label = '%s %.2f' % (names[int(cls)], conf)
    #                     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
    #
    #         # Print time (inference + NMS)
    #         print('%sDone. (%.3fs)' % (s, t2 - t1))
    #
    #         # Stream results
    #         if view_img:
    #             cv2.imshow(p, im0)
    #             if cv2.waitKey(1) == ord('q'):  # q to quit
    #                 raise StopIteration
    #
    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'images':
    #                 cv2.imwrite(save_path, im0)
    #             else:
    #                 if vid_path != save_path:  # new video
    #                     vid_path = save_path
    #                     if isinstance(vid_writer, cv2.VideoWriter):
    #                         vid_writer.release()  # release previous video writer
    #
    #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
    #                 vid_writer.write(im0)
    #
    # if save_txt or save_img:
    #     print('Results saved to %s' % os.getcwd() + os.sep + out)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + save_path)
    #
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument("--config-file", type=str, default="", metavar="FILE", help="path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    # parser.add_argument("--images_dir", type=str, default='demo', help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", type=str, default='demo/result', help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    # parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    # parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    # parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    # parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # opt = parser.parse_args()
    # opt.cfg = check_file(opt.cfg)  # check file
    # opt.names = check_file(opt.names)  # check file
    # print(opt)

    with torch.no_grad():
        detect(cfg=cfg,
                 ckpt=args.ckpt,
                 score_threshold=args.score_threshold,
                 output_dir=args.output_dir,
                 dataset_type=args.dataset_type)