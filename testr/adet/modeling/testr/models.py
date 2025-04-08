import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from testr.adet.layers.deformable_transformer import DeformableTransformer

from testr.adet.layers.pos_encoding import PositionalEncoding1D
from testr.adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from testr.adet.layers.pos_encoding import PositionalEncoding2D

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TESTR(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # self.backbone = backbone
        
        # fmt: off
        self.d_model                 = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead                   = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers      = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers      = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward         = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout                 = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation              = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels      = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points            = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points            = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals           = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale         = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points         = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes             = 1
        self.max_text_len            = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.voc_size                = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset          = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.text_pos_embed   = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on
        
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, 
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

                
        # if self.num_feature_levels > 1:
        #     strides = [8, 16, 32]
        #     # num_channels = [512, 1024, 2048]
        #     num_channels 
        #     num_backbone_outs = len(strides)
        #     input_proj_list = []
        #     for _ in range(num_backbone_outs):
        #         in_channels = num_channels[_]
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2d(in_channels, self.d_model, kernel_size=1),
        #             nn.GroupNorm(32, self.d_model),
        #         ))
        #     for _ in range(self.num_feature_levels - num_backbone_outs):
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2d(in_channels, self.d_model,
        #                       kernel_size=3, stride=2, padding=1),
        #             nn.GroupNorm(32, self.d_model),
        #         ))
        #         in_channels = self.d_model
        #     self.input_proj = nn.ModuleList(input_proj_list)
        # else:
        #     strides = [32]
        #     num_channels = [2048]
        #     self.input_proj = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(
        #                 num_channels[0], self.d_model, kernel_size=1),
        #             nn.GroupNorm(32, self.d_model),
        #         )])
        

        # JLP - extract feat channel
        # num_channels = [1280, 1280, 640, 320]
        # self.diff_feat_proj = nn.ModuleList([
        #         nn.Sequential(
        #         nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
        #         nn.GroupNorm(32, self.d_model),
        #         ),

        #         nn.Sequential(
        #         nn.Conv2d(num_channels[1], self.d_model, kernel_size=1),
        #         nn.GroupNorm(32, self.d_model),
        #         ),

        #         nn.Sequential(
        #         nn.Conv2d(num_channels[2], self.d_model, kernel_size=1),
        #         nn.GroupNorm(32, self.d_model),
        #         ),

        #         nn.Sequential(
        #         nn.Conv2d(num_channels[3], self.d_model, kernel_size=1),
        #         nn.GroupNorm(32, self.d_model),
        #         ),
        #     ]
        # )

        num_channels = [1280, 1280, 640, 320]
        self.diff_feat_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels[i], self.d_model, kernel_size=1),     # 1x1 projection
                nn.GroupNorm(32, self.d_model),
                nn.GELU(),

                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),  # 3x3 conv
                nn.GroupNorm(32, self.d_model),
                nn.GELU(),
            )
            for i in range(len(num_channels))
        ])

        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.diff_feat_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)

        # JLP 
        self.pos_enc_2d= PositionalEncoding2D(128, normalize=True)


    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_keypoints": The normalized keypoint coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # breakpoint()
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        # features, pos = self.backbone(samples)

        extracted_feats = samples
        pos = [self.pos_enc_2d(x) for x in extracted_feats]

        # if self.num_feature_levels == 1:    # f
        #     features = [features[-1]]
        #     pos = [pos[-1]]

        # srcs = []
        # masks = []
        # for l, feat in enumerate(features):     # len(features) = 3, extracted three features from resnet backbone
        #     src, mask = feat.decompose()
        #     srcs.append(self.input_proj[l](src))
        #     masks.append(mask)
        #     assert mask is not None
        # if self.num_feature_levels > len(srcs):
        #     _len_srcs = len(srcs)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](features[-1].tensors)
        #         else:
        #             src = self.input_proj[l](srcs[-1])
        #         m = masks[0]
        #         mask = F.interpolate(
        #             m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #         pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #         srcs.append(src)
        #         masks.append(mask)
        #         pos.append(pos_l)




        # (Pdb) extracted_feats[0].shape torch.Size([2, 1280, 14, 14])
        # (Pdb) extracted_feats[1].shape torch.Size([2, 1280, 28, 28])
        # (Pdb) extracted_feats[2].shape torch.Size([2, 640, 56, 56])
        # (Pdb) extracted_feats[3].shape torch.Size([2, 320, 56, 56])
        
        srcs = []
        masks = []
        for l, feat in enumerate(extracted_feats):
            b, _, feat_H, feat_W = feat.shape
            srcs.append(self.diff_feat_proj[l](feat))
            masks.append(torch.zeros(b, feat_H, feat_W).to(bool).to(feat.device))


        # (Pdb) srcs[0].shape torch.Size([2, 256, 14, 14])
        # (Pdb) srcs[1].shape torch.Size([2, 256, 28, 28])
        # (Pdb) srcs[2].shape torch.Size([2, 256, 56, 56])
        # (Pdb) srcs[3].shape torch.Size([2, 256, 56, 56])

        # breakpoint()


        ''' 이해 O
        우리는 bbox, 또는 polygon, 또는 word가 되는 query를 learn하고 싶음.
        따라서 learnable query를 선언해주고 이것이 model을 거치면서 bbox 또는 polygon 또는 word가 됨.
        이제 query 하나의 shape를 결정짓는 요소는 box 또는 polygon 또는 word를 어떻게 표현할지에 따라서 다름.

        box로 query를 표현할 경우, box는 (x,y,w,h) 4개의 값으로 표현하면 되기에, 하나의 query.shape = (4,) 이면 된다.
        N개의 query를 사용할 경우 -> (N,4) 이면 충분

        polygon을 query로 표현할 경우, polygon은 16개의 point, 각 point는 2개의 좌표가 필요하기에, query.shape = (16,2) 가 된다.
        N개의 query를 사용할 경우 -> (N,16,2)이면 충분
        
        word를 query로 표현할 경우, word의 maxlen=20이라고 하면, 하나의 자리에 최대 26개의 알파벳중 하나가 오기에, query.shape = (20, 26) 가 된다.
        N개의 query를 사용할 경우 -> (N,20,26) 이면 충분

        
        다시 돌아와서, query가 box,polygon,word등 다양하게 될 수 있다. 이는 최종 output linear layer을 거쳐서 dimension을 맞춰줄 수 있다.
        예를 들어 box가 될 query의 경우, (N,256) -> linear -> (N,4) 로 마지막에 query별로 4개의 box좌표값만 뱉으면 된다.

        따라서 선언할 때는 굳이 최종 shape으로 선언해줄 필요가 없다.
        box의 경우, torch.rand(hidden_dim) -> linear -> 4

        polygon의 경우 torch.rand(16, hidden_dim) -> linear -> (16,2)
        or
        poly_emb = nn.Embedding(16,hidden_dim)
        poly_emb.weight: (16,256) -> linear -> (16,2)
        
        word의 경우, torch.rand(20, hidden_dim) -> linear -> (20,26) 
        or
        word_emb = nn.Embedding(20,hidden_dim)
        word_emb.weight: (20,hidden_dim) -> linear -> (20,26)


        '''

        # breakpoint()

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)                     # 100 16 256
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)        # 100 25 256
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)                                 # 100 25 256

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, enc_box_ref_point = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)

        # breakpoint()
        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        # breakpoint()
        outputs_class = torch.stack(outputs_classes)    # 6 b 100 16 1
        outputs_coord = torch.stack(outputs_coords)     # 6 b 100 16 2
        outputs_text = torch.stack(outputs_texts)       # 6 b 100 25 97

        # breakpoint()
        out = {'pred_logits': outputs_class[-1],        # b 100 16 1
               'pred_ctrl_points': outputs_coord[-1],   # b 100 16 2
               'pred_texts': outputs_text[-1]}          # b 100 25 97
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, 'pred_filtered_boxes': enc_box_ref_point}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]