# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve() # 得到绝对路径 C:\Users\zkh\Desktop\yolov5\detect.py
ROOT = FILE.parents[0]  # 获得detect.py的父目录  YOLOv5 root directory, C:\Users\zkh\Desktop\yolov5
if str(ROOT) not in sys.path: # 模块的查询路径的列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ,绝对路径转换成相对路径

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)  模型权重
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source) # 'python detect.py --source data\\images\\bus.jpg'
    save_img = not nosave and not source.endswith('.txt')  # save inference images  是否保存推理的图像 为 True 
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)   # suffix表示后缀
    # 使用Path(source).suffix[1:]获取source路径的后缀名，并将其去除第一个字符（通常是.）。
    # 然后，检查该后缀名是否在IMG_FORMATS和VID_FORMATS列表中。
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # False
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) # False
    if is_url and is_file:
        source = check_file(source)  # download 下载图片或者视频
 


    # Directories 新建一个保存结果的文件夹,project和name就是project=ROOT / 'runs/detect',和 name='exp', 
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run, runs\\detect\\exp3
    # 将project和name拼接成一个路径。Path(project)创建一个Path对象,表示project的路径,然后使用/运算符将其与name拼接。
 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    """
    save_dir 是一个变量，表示将要保存文件的目录。
    save_txt 是一个布尔变量，用于指示是否保存为文本文件。
    如果 save_txt 为真，则将在 save_dir 目录下创建名为 'labels' 的子目录。
    如果 save_txt 为假，则直接在 save_dir 目录下创建文件夹。
    mkdir() 是一个函数，用于创建目录。
    parents = True 表示如果父目录不存在，也会一并创建父目录。
    exist_ok=True 表示如果目录已经存在，不会引发错误，而是继续执行。
    """



    # Load model
    device = select_device(device)

    # 根据使用的深度学习框架记载模型（pytorch tensorflow ）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt  # stride = 32, pt是pytorch的缩写，stride等于321
    # print("pt",pt)  result:True
    # print("names",names)  打印出所有的类别名，一共有80个类别

    imgsz = check_img_size(imgsz, s=stride)  # check image 图像尺寸size必须是32的倍数

    # Dataloader
    if webcam:
        view_img = check_imshow()  #检查能否显示图片

        # cudnn.benchmark 标志用于启用 CUDNN 库的自动调优功能。
        # 这允许 CUDNN 自动调整其内部算法,找到最快的卷积算法,适合您的硬件。这可以带来显著的加速效果,特别是对于输入大小保持不变的模型。
        cudnn.benchmark = True  # set True to speed up constant image size inference
        """
        将cudnn.benchmark设置为True可以加速PyTorch中对具有恒定输入尺寸的模型进行推断的过程。下面是它的作用和工作原理;

        CuDNN(CUDA深度神经网络库)是一种针对深度学习任务进行优化的GPU加速库。
        在PyTorch中,CuDNN主要用于执行卷积和池化等操作,以加速神经网络的训练和推断过程。
        当cudnn.benchmark设置为True时,PyTorch会在每次进行卷积操作时,根据当前的输入尺寸和其他相关参数，自动选择最适合当前硬件环境的卷积实现策略。
        这个选择过程会在第一次执行卷积操作时进行，并且会根据硬件、输入尺寸和其他条件的变化而重新选择。

        ，以找到最优的卷积算法。然后，它会将这些选择缓存起来，以便在后续的推断中重复使用。
        这样一来，当使用相同尺寸的输入进行推断时,PyTorch就可以直接使用之前缓存的最优算法,从而提高推断速度。
        """
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup 对模型进行预热，为后续的推理做准备。
    dt, seen = [0.0, 0.0, 0.0], 0  # 如时间记录和已处理数据的计数。
    for path, im, im0s, vid_cap, s in dataset: # "C:\\Users\\zkh\\Desktop\\yolov5\\data\\images\\bus.jpg"
        t1 = time_sync()
        im = torch.from_numpy(im).to(device) # torch.Size([3, 640, 480])
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0 将图像的像素值从 0 到 255 的范围归一化到 0.0 到 1.0 的范围。
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim , torch.Size([1, 3, 640, 480])
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 根据是否需要可视化来确定一个路径，并可能创建相关的目录。
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        # 使用模型对输入的图像 im 进行推理计算，同时考虑是否进行数据增强以及是否进行可视化操作，得到预测结果 pred 。
        pred = model(im, augment=augment, visualize=visualize) # torch.Size([1, 18900, 85])  检测出18900个检测框
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # 1,5,6 . [6.72000e+02, 3.95000e+02, 8.10000e+02, 8.78000e+02, 8.96172e-01, 0.00000e+00]
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image,torch.Size([5, 6])   det表示5个检测框  以及对应的6个信息
            seen += 1  # 每处理一张图片  数量会加1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)   # 如果没有frame属性  这个值就是0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, "runs\\detect\\exp3","bus.jpg"
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string 图片的高度和宽度，作为打印信息
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results

                # 获取检测结果中最后一列（通常是类别标识）的不同值
                for c in det[:, -1].unique():
                    # 计算属于该类别的检测结果的数量。
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # *xyxy表示坐标，conf表示置信度，cls表示类别
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results #返回画好的图片
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(10)  # 10 millisecond

            # Save results (image with detections) 保存图像
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()  #创建参数解析器对象
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    """
    nargs参数用于指定应该从命令行中接受多少个参数。
    当nargs的值为'+'时，表示接受一个或多个参数，并将这些参数作为列表存储。
    例如，如果在命令行中指定了--weights a.pt b.pt c.pt,那么args.weights将存储为['a.pt', 'b.pt', 'c.pt']。
    --weights是一个选项参数，用于指定模型的路径。nargs='+'表示可以接受一个或多个模型路径，
    然后将这些路径作为一个列表存储在args.weights中。
    如果在命令行中没有指定--weights参数,则默认值为ROOT / 'yolov5s.pt';
    help参数是用于提供关于该参数的帮助信息,当用户在命令行中使用--help选项时.这些帮助信息将被显示出来，以便用户了解如何正确使用该参数。
    在这个例子中，帮助信息是 'model path(s)'，用于描述--weights参数表示模型路径的含义。
    """

    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    """
    该代码添加了一个名为 --source 的选项参数。它用于指定文件、目录、URL或通配符,或者使用数字 0 表示使用网络摄像头作为输入。

    --source:指定了参数的名称为 --source。
    type=str:指定了参数的类型为字符串，即输入的值将被解析为字符串。
    default=ROOT / 'data/images'：如果在命令行中没有指定 --source 参数，则默认值为 ROOT / 'data/images'。
    这里 ROOT / 'data/images' 表示文件系统中的路径，具体的值取决于程序中定义的 ROOT 变量。
    help='file/dir/URL/glob, 0 for webcam'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --source 参数的用途,即它可以接受文件、目录、URL、通配符或数字 0 表示使用网络摄像头作为输入。
    
    """
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    """
    该代码添加了一个名为 --data 的选项参数，用于指定数据集的路径。

    --data:指定了参数的名称为 --data。
    type=str:指定了参数的类型为字符串，即输入的值将被解析为字符串。
    default=ROOT / 'data/coco128.yaml'：如果在命令行中没有指定 --data 参数，则默认值为 ROOT / 'data/coco128.yaml'。
    这里 ROOT / 'data/coco128.yaml' 表示文件系统中的路径，具体的值取决于程序中定义的 ROOT 变量。
    help='(optional) dataset.yaml path'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --data 参数的用途，即它用于指定数据集的路径，这是一个可选的参数。
    
    """

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    """
    该代码添加了一个名为 --imgsz 的选项参数，同时还添加了 --img 和 --img-size 作为该选项参数的别名。该选项参数用于指定推断过程中的图像尺寸。
    --imgsz、--img、--img-size:指定了参数的名称为 --imgsz,同时添加了 --img 和 --img-size 作为该选项参数的别名。
    这意味着可以在命令行中使用任意一个名称来指定该选项参数。
    nargs='+'：指定该选项参数可以接受一个或多个值，并将这些值作为列表存储。
    例如，在命令行中指定 --imgsz 640 480 将会将 [640, 480] 存储在参数的值中。
    type=int:指定了参数的类型为整数，即输入的值将被解析为整数。
    default=[640]：如果在命令行中没有指定该选项参数，则默认值为 [640]，即一个包含单个整数 640 的列表。
    help='inference size h,w'：提供了关于该选项参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了该选项参数的用途，即指定推断过程中的图像尺寸，其中 h 和 w 表示高度和宽度。
        
    """

    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    """
    该代码添加了一个名为 --conf-thres 的选项参数，用于指定置信度阈值。

    --conf-thres:指定了参数的名称为 --conf-thres。
    type=float:指定了参数的类型为浮点数，即输入的值将被解析为浮点数。
    default=0.25:如果在命令行中没有指定 --conf-thres 参数，则默认值为 0.25。
    help='confidence threshold'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --conf-thres 参数的用途，即指定置信度阈值。
    """

    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')

    """
    该代码添加了一个名为 --max-det 的选项参数，用于指定每张图像的最大检测数。

    --max-det:指定了参数的名称为 --max-det。
    type=int:指定了参数的类型为整数，即输入的值将被解析为整数。
    default=1000:如果在命令行中没有指定 --max-det 参数，则默认值为 1000。
    help='maximum detections per image'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --max-det 参数的用途，即指定每张图像的最大检测数。
    """
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    """
    该代码添加了一个名为 --device 的选项参数,用于指定使用的计算设备(GPU 或 CPU)。

    --device:指定了参数的名称为 --device。
    default=''：如果在命令行中没有指定 --device 参数，则默认值为空字符串 ''。
    help='cuda device, i.e. 0 or 0,1,2,3 or cpu'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --device 参数的用途，即指定使用的计算设备。用户可以输入一个或多个 GPU 设备的索引（例如 0 或 0,1,2,3),也可以输入 cpu 表示使用 CPU。
    """
    parser.add_argument('--view-img', action='store_true', help='show results')  #会闪现
    """
    该代码添加了一个名为 --view-img 的选项参数，用于控制是否显示结果。

    --view-img:指定了参数的名称为 --view-img。
    action='store_true'：指定当命令行中出现 --view-img 参数时，将该参数设为 True。如果命令行中没有出现该参数,则默认为 False。
    help='show results'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --view-img 参数的用途，即控制是否显示结果。当指定了 --view-img 参数时，程序将显示结果；否则，结果将不会被显示。
    """
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')

    """
    该代码添加了一个名为 --save-txt 的选项参数，用于控制是否将结果保存为文本文件。
    --save-txt:指定了参数的名称为 --save-txt。
    action='store_true'：指定当命令行中出现 --save-txt 参数时，将该参数设为 True。如果命令行中没有出现该参数,则默认为 False。
    help='save results to *.txt'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --save-txt 参数的用途，即控制是否将结果保存为文本文件。
    当指定了 --save-txt 参数时，程序将把结果保存为文本文件；否则，结果将不会保存为文本文件。

    """

    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    """
    用于控制是否在保存的文本标签中包含置信度信息。
    --save-conf:指定了参数的名称为 --save-conf。
    action='store_true'：指定当命令行中出现 --save-conf 参数时，将该参数设为 True。如果命令行中没有出现该参数,则默认为 False。
    help='save confidences in --save-txt labels'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --save-conf 参数的用途，即控制是否在保存的文本标签中包含置信度信息。
    当指定了 --save-conf 参数时，程序将在保存的文本标签中包含置信度信息；否则，文本标签中将不包含置信度信息。

    """
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    """
    该代码添加了一个名为 --save-crop 的选项参数，用于控制是否保存裁剪后的预测框。
    --save-crop:指定了参数的名称为 --save-crop。
    action='store_true'：指定当命令行中出现 --save-crop 参数时，将该参数设为 True。
    如果命令行中没有出现该参数，则默认为 False。
    help='save cropped prediction boxes'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --save-crop 参数的用途，即控制是否保存裁剪后的预测框。
    当指定了 --save-crop 参数时，程序将保存裁剪后的预测框；否则，不保存裁剪后的预测框。
    """
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    """
    该代码添加了一个名为 --classes 的选项参数，用于按类别进行过滤。
    --classes：指定了参数的名称为 --classes。
    nargs='+'：指定该选项参数可以接受一个或多个值，并将这些值作为列表存储。例如，在命令行中指定 --classes 0 2 3 将会将 [0, 2, 3] 存储在参数的值中。
    type=int：指定了参数的类型为整数，即输入的值将被解析为整数。
    help='filter by class: --classes 0, or --classes 0 2 3'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --classes 参数的用途，即按照指定的类别进行过滤。
    用户可以在命令行中使用一个或多个整数值来指定过滤的类别。
    例如，--classes 0 表示只保留类别为 0 的结果，--classes 0 2 3 表示只保留类别为 0、2 和 3 的结果。
    """

    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    """
    用于执行类别不可知的非极大值抑制(class-agnostic NMS)。
    --agnostic-nms:指定了参数的名称为 --agnostic-nms。
    action='store_true'：指定当命令行中出现 --agnostic-nms 参数时，将该参数设为 True。如果命令行中没有出现该参数，则默认为 False。
    help='class-agnostic NMS'：提供了关于该参数的帮助信息，当用户使用 --help 选项时会显示。
    帮助信息描述了 --agnostic-nms 参数的用途，即执行类别不可知的非极大值抑制。
    当指定了 --agnostic-nms 参数时，程序将在执行非极大值抑制时不考虑物体的类别信息；否则，会考虑物体的类别信息进行抑制。
    """
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # 即执行增强推理。当指定了 --augment 参数时，程序将使用增强技术（如随机裁剪、缩放、翻转等）对输入图像进行增强，以提升模型的性能和鲁棒性。
 
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 即可视化特征。当指定了 --visualize 参数时，程序将可视化模型的特征，以帮助理解模型的工作方式、观察特征的变化等。

    parser.add_argument('--update', action='store_true', help='update all models')
    # 更新所有模型。当指定了 --update 参数时，程序将执行更新操作，可能包括下载最新的模型权重文件、更新配置文件等。


    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 即指定结果保存的项目路径。用户可以在命令行中使用 --project 参数来指定结果保存的项目路径，如果没有指定，则使用默认值。结果将保存在指定的项目路径下。

    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 指定结果保存的名称。用户可以在命令行中使用 --name 参数来指定结果保存的名称，如果没有指定，则使用默认值。结果将保存在项目路径下，并以指定的名称命名。


    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') #没啥用
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # 用户可以在命令行中使用 --line-thickness 参数来指定边界框线条的粗细，如果没有指定，则使用默认值。边界框线条的粗细以像素为单位。

    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    # 这行代码将解析命令行参数，并将结果存储在变量 opt 中。opt 是一个包含解析后的参数值的对象。
    
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt)) # 这行代码将使用 vars() 函数获取 opt 对象的属性和值，
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop')) # exclude参数可以使用任何字符串值作为参数值，用于指定要排除的依赖项或模块。
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
