# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build 在构建过程中计算的步长
    onnx_dynamic = False  # ONNX export parameter ONNX 导出参数  表示是否使用动态计算
    export = False  # export mode 导出模式 表示是否处于导出模式


    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer 检测层
        super().__init__()  # 调用父类的初始化方法
        self.nc = nc  # number of classes  80
        self.no = nc + 5  # number of outputs per anchor  85
        self.nl = len(anchors)  # number of detection layers  3  检测层的数量,三个检测头
        self.na = len(anchors[0]) // 2  # number of anchors  3 锚点的数量，表示每个检测层中使用的锚点数量，这里是3个
        self.grid = [torch.zeros(1)] * self.nl  # init grid  
        # 初始化网格，这里创建了一个长度为 nl 的列表，每个元素都是一个形状为 (1,) 的全零张量。


        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 初始化锚点网格，与 grid 类似，创建了一个长度为 nl 的列表，每个元素都是一个形状为 (1,) 的全零张量。

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 注册缓冲区 anchors，其中存储了锚点的张量，将输入的 anchors 转换为浮点型张量，并将其形状调整为 (nl, na, 2)。

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output 初始化一个空列表 z，用于存储推断输出

        # self.nl 等于 3
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv  使用第 i 个检测层的卷积模块 self.m[i] 对输入特征图 x[i] 进行卷积操作。

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 提取经过检测头处理过的特征图 x[i] 的批次大小（bs）、通道数（用 _ 忽略）、高度（ny）和宽度（nx）。

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # 对特征图 x[i] 进行形状调整，从 (bs, 255, 20, 20) 转换为 (bs, 3, 20, 20, 85)，并进行维度的置换。

            if not self.training:  # inference 检查模型是否处于推理状态，不是训练状态

                # 判断是否需要更新网格信息。
                # 如果设置了 self.onnx_dynamic 或者当前的网格大小与特征图 x[i] 的大小不一致，则调用 _make_grid(nx, ny, i) 方法更新网格信息。
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # 检查是否在原地进行操作
                y = x[i].sigmoid()  # 对卷积结果 x[i] 应用 sigmoid 激活函数。
                if self.inplace:
                    # 预测结果的前两个通道（xy坐标）进行处理，包括乘以2、加上网格信息、乘以步长。
                    """
                    这行代码的作用是将预测结果中的前两个通道（xy坐标）进行缩放、平移和应用步长的操作。
                    这是为了将预测的相对坐标转换为相对于整个图像的绝对坐标，并考虑到模型的步长和网格信息。
                    """
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy

                    """
                    这行代码的作用是对预测结果中的后两个通道（宽高）进行缩放、平方和乘以锚框尺寸的操作。
                    这是为了将预测的相对宽高转换为相对于整个图像的绝对宽高，并考虑到模型的锚框尺寸。
                    """
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()

        # 判断是不是字典类型，由于不是字典类型，因此会执行
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name # 获得文件名
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict 读取ymal文件中的内容，以python 字典的形式存储

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels, 3


        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist:[4, 6, 10, 14, 17, 20, 23]
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward：[8,16,32]
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)  # 在使用anchor是是在最终的预测特征层上使用的。
            self.stride = m.stride
            self._initialize_biases()  # only run once  初始化参数

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict:yolov5s.yaml, input_channels(3):[3]
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'] # gd:0.33, gw:0.5
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors anchor的数量 3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) = 255   模型最终的输出通道数  3*（80+5） = 255  5 = 4 + 1  （4是坐标，1是置信度）  

    # 每个操作（卷积或C3）都有输入和输出，输入用c1表示，输出用c2表示
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out（save用于统计哪些层需要保存，用于cancat操作，）


    # n 是module number  模块的个数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # 以第一层为例： from:-1, number:1, module:'Conv', args:[64, 6, 2, 2]  第0个参数是64
        m = eval(m) if isinstance(m, str) else m  # eval strings, m:<class 'models.common.Conv'>  将字符串转为common.py中的类
        for j, a in enumerate(args): # j是 args:[64, 6, 2, 2]中每个元素的索引，a是对应索引的值
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings,最后转为： [64, 6, 2, 2] 列表， 每个元素是int类型
            except NameError:
                pass
        # # gd:0.33, gw:0.5
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain 计算出有多少个C3
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost):
            
            #  c1表示输入通道数，c2表示输出的通道数  ch是传进来的input channel = [3] 只有一个元素  此处ch[f] = 3
            # args[0] 表示64
            c1, c2 = ch[f], args[0] # c1:3, c2:64  gw:0.5  gw是通道倍数
            if c2 != no:  # if not output  判断该输出通道和最终的输出通道255是否相等
                c2 = make_divisible(c2 * gw, 8) # c2= 64*0.5 = 32   并判断是不是8的倍数，如果不是会变成8的倍数

            args = [c1, c2, *args[1:]] # args[3, 32, 6, 2, 2]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats 多少个C3数值插入输入通道数c1和输出通道数c2的后面
                n = 1  # 再另n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]


        """
        m 表示 module
        如果 n > 1,那么就创建一个包含 n 个相同模块的 nn.Sequential 对象。
        每个模块都是通过调用 m(*args) 创建的。
        如果 n == 1,那么就直接返回 m(*args) 创建的单个模块。
        """
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  #创建 module 序列（这一层创建出来）
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params 统计这一层的参数量np表示参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type,  number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print打印每一层的信息

        # 记录一下哪些层需要保存
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2) # [32],[32,64],[32,64,64]  将每一层的输出通道加入到ch列表中
    return nn.Sequential(*layers), sorted(save) # [6, 4, 14, 10, 17, 20, 23] -> [4, 6, 10, 14, 17, 20, 23]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args() # Namespace(batch_size=1, cfg='yolov5s.yaml', device='', line_profile=False, profile=False, test=False)
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        _ = model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
