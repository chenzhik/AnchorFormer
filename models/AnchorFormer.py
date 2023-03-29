# -*- encoding: utf-8 -*-
import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.expansion_penalty.expansion_penalty_module import expansionPenaltyModule
from .Transformer import AnchorTransformer
from .Morphing import PointMorphing


from .build import MODELS

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc    

# 3D completion
@MODELS.register_module()
class AnchorFormer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.num_encoder_blk = config.num_encoder_blk # 6
        self.num_decoder_blk = config.num_decoder_blk # 8
        self.sparse_expansion_lambda = config.sparse_expansion_lambda
        self.dense_expansion_lambda = config.dense_expansion_lambda

        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = AnchorTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [self.num_encoder_blk, self.num_decoder_blk], num_query = self.num_query)
        self.upsample_net = PointMorphing(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point 

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map_global = nn.Linear(1024, self.trans_dim)
        self.reduce_map_local = nn.Linear(self.trans_dim + 3, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.penalty_func = expansionPenaltyModule()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine
    
    def get_penalty(self):
        dist, _, mean_mst_dis = self.penalty_func(self.pred_coarse_point, 16, self.sparse_expansion_lambda)
        dist_dense, _, mean_mst_dis = self.penalty_func(self.pred_dense_point, 64, self.dense_expansion_lambda)
        loss_mst = torch.mean(dist) 
        loss_mst_fine = torch.mean(dist_dense) 
        return loss_mst, loss_mst_fine
    
    def forward(self, xyz):
        q, coarse_point_cloud = self.base_model(xyz) # B M C and B M 3

        self.pred_coarse_point = coarse_point_cloud
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        
        rebuild_feature = torch.cat([global_feature.unsqueeze(-2).expand(-1, M, -1), q, coarse_point_cloud], dim=-1)  # B M 1027 + C
        global_feature = rebuild_feature[:,:,:1024].reshape(B*M,1024)
        local_feature = rebuild_feature[:,:,1024:].reshape(B*M,self.trans_dim+3)
        global_feature = self.reduce_map_global(global_feature)
        local_feature = self.reduce_map_local(local_feature)
        
        relative_xyz = self.upsample_net(global_feature, local_feature).reshape(B, M, 3, -1)    # B M 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3
        
        self.pred_dense_point = rebuild_points
    
        inp_sparse = fps(xyz, self.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        ret = (coarse_point_cloud, rebuild_points)
        return ret
