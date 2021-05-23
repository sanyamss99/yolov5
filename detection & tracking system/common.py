# This file contains modules common to various models

import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def forward(self, imgs, size=640, augment=False, profile=False):
        # supports inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   opencv:     imgs = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:        imgs = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:      imgs = np.zeros((720,1280,3))  # HWC
        #   torch:      imgs = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:   imgs = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        if not isinstance(imgs, list):
            imgs = [imgs]
        shape0, shape1 = [], []  # image and inference shapes
        batch = range(len(imgs))  # batch size
        for i in batch:
            imgs[i] = np.array(imgs[i])  # to numpy
            if imgs[i].shape[0] < 5:  # image in CHW
                imgs[i] = imgs[i].transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            imgs[i] = imgs[i][:, :, :3] if imgs[i].ndim == 3 else np.tile(imgs[i][:, :, None], 3)  # enforce 3ch input
            s = imgs[i].shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(imgs[i], new_shape=shape1, auto=False)[0] for i in batch]  # pad
        x = np.stack(x, 0) if batch[-1] else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in batch:
            if y[i] is not None:
                y[i][:, :4] = scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if show:
                img.show(f'Image {i}')  # show
            if pprint:
                print(str)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


#encoder decoder attention layer


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)



class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

