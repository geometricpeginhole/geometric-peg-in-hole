import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import p3d_transforms
import geometric_peg_in_hole.pytorch3d_transforms as p3d_transforms
import matplotlib.pyplot as plt


class BCModel(nn.Module):
    def __init__(self, image_encoder, action_decoder, rotation_type, rotation_loss_type, fusion_type, proprioception, resize=224) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.action_decoder = action_decoder
        self.resize = resize
        self.rotation_type = rotation_type
        self.rotation_loss_type = rotation_loss_type
        self.fusion_type = fusion_type
        self.proprioception = proprioception
    
    def forward(self, image, robot_state):
        B, O, V, C, H, W = image.shape
        image = torch.reshape(image, (B * O * V, C, H, W))
        image = image[..., :min(H, W), :min(H, W)]
        image = TF.resize(image, self.resize)
        image = torch.reshape(image, (B, O, V, C, self.resize, self.resize))
        # V = len(image)
        # B, O, C, _, _ = image.shape
        # image = [torch.reshape(view, (-1, view.shape[-3], view.shape[-2], view.shape[-1])) for view in image]
        # # V, [B * O, C, H, W]
        # # image = [TF.center_crop(view, min(view.shape[-1], view.shape[-2])) for view in image]
        # image = [view[..., :min(view.shape[-1], view.shape[-2]), :min(view.shape[-1], view.shape[-2])] for i_view, view in enumerate(image)]
        # # V, [B * O, C, H, H]
        # image = [TF.resize(view, self.resize) for view in image]
        # # V, [B * O, C, resize, resize]
        # image = torch.stack(image, dim=1)
        # # B * O, V, C, resize, resize
        # image = torch.reshape(image, (B, O, V, C, image.shape[-2], image.shape[-1]))
        # # B, O, V, C, resize, resize

        B, O, V, C, H, W = image.shape
        x = torch.reshape(image, (-1, C, H, W))
        # x = x[..., :min(H, W), :min(H, W)]
        # x = TF.resize(x, self.resize)
        # B * O * V, C, resize, resize

        x = self.image_encoder(x)
        # B * O * V, D, 1, 1
        if self.fusion_type == 'concat':
            x = torch.reshape(x, (B, O, -1))
            # B, O, V * D
        elif self.fusion_type == 'add':
            x = torch.reshape(x, (B, O, V, -1))
            x = torch.sum(x, dim=2)
            # B, O, D
            
        # B, O, V * D
        if self.proprioception == 'with_prop':
            # print(x.shape, robot_state.shape)
            x = torch.cat((x, robot_state), dim=-1)
        # B, O, V * D + 14
        x = self.action_decoder(x)
        # B, A, 14
        return x
    
    def get_action(self, image, robot_state):
        if self.rotation_type == 'quat':
            x = self.forward(image, robot_state)
            return x
        elif self.rotation_type == '6d':
            x = self.forward(image, robot_state)
            pred_arm1_pos = x[:, :, :3]
            pred_arm1_rotx = x[:, :, 3:6]
            pred_arm1_roty = x[:, :, 6:9]
            pred_arm2_pos = x[:, :, 9:12]
            pred_arm2_rotx = x[:, :, 12:15]
            pred_arm2_roty = x[:, :, 15:18]
            pred_arm1_rotx = F.normalize(pred_arm1_rotx, dim=-1)
            pred_arm1_roty = F.normalize(pred_arm1_roty - pred_arm1_rotx * torch.sum(pred_arm1_rotx * pred_arm1_roty, dim=-1, keepdim=True), dim=-1)
            pred_arm2_rotx = F.normalize(pred_arm2_rotx, dim=-1)
            pred_arm2_roty = F.normalize(pred_arm2_roty - pred_arm2_rotx * torch.sum(pred_arm2_rotx * pred_arm2_roty, dim=-1, keepdim=True), dim=-1)
            pred_arm1_rotz = torch.cross(pred_arm1_rotx, pred_arm1_roty, dim=-1)
            pred_arm2_rotz = torch.cross(pred_arm2_rotx, pred_arm2_roty, dim=-1)
            pred_arm1_mat = torch.stack((pred_arm1_rotx, pred_arm1_roty, pred_arm1_rotz), dim=-1)
            pred_arm2_mat = torch.stack((pred_arm2_rotx, pred_arm2_roty, pred_arm2_rotz), dim=-1)
            pred_arm1_rot = p3d_transforms.matrix_to_quaternion(pred_arm1_mat)
            pred_arm2_rot = p3d_transforms.matrix_to_quaternion(pred_arm2_mat)
            x = torch.cat((pred_arm1_pos, pred_arm1_rot, pred_arm2_pos, pred_arm2_rot), dim=-1)
            return x
            
    
    def loss(self, output, label):
        losses = {}
        if self.rotation_type == 'quat':
            pred_arm1_pos = output[:, :, :3]
            pred_arm1_rot = output[:, :, 3:7]
            pred_arm2_pos = output[:, :, 7:10]
            pred_arm2_rot = output[:, :, 10:14]
            pred_arm1_rot = F.normalize(pred_arm1_rot, dim=-1)
            pred_arm2_rot = F.normalize(pred_arm2_rot, dim=-1)
            label_arm1_pos = label[:, :, :3]
            label_arm1_rot = label[:, :, 3:7]
            label_arm2_pos = label[:, :, 7:10]
            label_arm2_rot = label[:, :, 10:14]
            pred_arm1_mat = p3d_transforms.quaternion_to_matrix(pred_arm1_rot)
            pred_arm2_mat = p3d_transforms.quaternion_to_matrix(pred_arm2_rot)
            label_arm1_mat = p3d_transforms.quaternion_to_matrix(label_arm1_rot)
            label_arm2_mat = p3d_transforms.quaternion_to_matrix(label_arm2_rot)
        elif self.rotation_type == '6d':
            pred_arm1_pos = output[:, :, :3]
            pred_arm1_rot = output[:, :, 3:9]
            pred_arm2_pos = output[:, :, 9:12]
            pred_arm2_rot = output[:, :, 12:18]
            label_arm1_pos = label[:, :, :3]
            label_arm1_rot = label[:, :, 3:9]
            label_arm2_pos = label[:, :, 9:12]
            label_arm2_rot = label[:, :, 12:18]
            pred_arm1_rotx = pred_arm1_rot[:, :, :3]
            pred_arm1_roty = pred_arm1_rot[:, :, 3:6]
            pred_arm2_rotx = pred_arm2_rot[:, :, :3]
            pred_arm2_roty = pred_arm2_rot[:, :, 3:6]
            pred_arm1_rotx = F.normalize(pred_arm1_rotx, dim=-1)
            pred_arm1_roty = F.normalize(pred_arm1_roty - pred_arm1_rotx * torch.sum(pred_arm1_rotx * pred_arm1_roty, dim=-1, keepdim=True), dim=-1)
            pred_arm2_rotx = F.normalize(pred_arm2_rotx, dim=-1)
            pred_arm2_roty = F.normalize(pred_arm2_roty - pred_arm2_rotx * torch.sum(pred_arm2_rotx * pred_arm2_roty, dim=-1, keepdim=True), dim=-1)
            pred_arm1_rotz = torch.cross(pred_arm1_rotx, pred_arm1_roty, dim=-1)
            pred_arm2_rotz = torch.cross(pred_arm2_rotx, pred_arm2_roty, dim=-1)
            pred_arm1_mat = torch.stack((pred_arm1_rotx, pred_arm1_roty, pred_arm1_rotz), dim=-1)
            pred_arm2_mat = torch.stack((pred_arm2_rotx, pred_arm2_roty, pred_arm2_rotz), dim=-1)
            label_arm1_mat = torch.stack((label_arm1_rot[:, :, :3], label_arm1_rot[:, :, 3:6],
                                          torch.cross(label_arm1_rot[:, :, :3], label_arm1_rot[:, :, 3:6])), dim=-1)
            label_arm2_mat = torch.stack((label_arm2_rot[:, :, :3], label_arm2_rot[:, :, 3:6],
                                          torch.cross(label_arm2_rot[:, :, :3], label_arm2_rot[:, :, 3:6])), dim=-1)
            
            losses['rotx_loss'] = ((1 - torch.sum(pred_arm1_rotx * label_arm1_rot[:, :, :3], dim=-1)).mean()\
                                    + (1 - torch.sum(pred_arm2_rotx * label_arm2_rot[:, :, :3], dim=-1)).mean()) / 2
            losses['roty_loss'] = ((1 - torch.sum(pred_arm1_roty * label_arm1_rot[:, :, 3:6], dim=-1)).mean()\
                                    + (1 - torch.sum(pred_arm2_roty * label_arm2_rot[:, :, 3:6], dim=-1)).mean()) / 2
            losses['rotz_loss'] = ((1 - torch.sum(pred_arm1_rotz * label_arm1_mat[:, :, 2], dim=-1)).mean()\
                                    + (1 - torch.sum(pred_arm2_rotz * label_arm2_mat[:, :, 2], dim=-1)).mean()) / 2

        losses['x_loss'] = (F.mse_loss(pred_arm1_pos[..., 0], label_arm1_pos[..., 0])\
                            + F.mse_loss(pred_arm2_pos[..., 0], label_arm2_pos[..., 0])) / 2
        losses['y_loss'] = (F.mse_loss(pred_arm1_pos[..., 1], label_arm1_pos[..., 1])\
                            + F.mse_loss(pred_arm2_pos[..., 1], label_arm2_pos[..., 1])) / 2
        losses['z_loss'] = (F.mse_loss(pred_arm1_pos[..., 2], label_arm1_pos[..., 2])\
                            + F.mse_loss(pred_arm2_pos[..., 2], label_arm2_pos[..., 2])) / 2
        losses['pos_loss'] = (losses['x_loss'] + losses['y_loss'] + losses['z_loss']) / 3
        losses['rot_mse_loss'] = (F.mse_loss(pred_arm1_rot, label_arm1_rot) + F.mse_loss(pred_arm2_rot, label_arm2_rot)) / 2
        losses['rot_frob_loss'] = (F.mse_loss(pred_arm1_mat, label_arm1_mat) \
                                   + F.mse_loss(pred_arm2_mat, label_arm2_mat)) / 2
        losses['rot_deg_loss'] = (torch.norm(p3d_transforms.matrix_to_axis_angle(pred_arm1_mat @ label_arm1_mat.transpose(-1, -2)), dim=-1).mean() * 180 / torch.pi \
                                 + torch.norm(p3d_transforms.matrix_to_axis_angle(pred_arm2_mat @ label_arm2_mat.transpose(-1, -2)), dim=-1).mean() * 180 / torch.pi)
        
        if self.rotation_loss_type == 'mse':
            losses['rot_loss'] = losses['rot_mse_loss']
        elif self.rotation_loss_type == 'frob':
            losses['rot_loss'] = losses['rot_frob_loss']
        elif self.rotation_loss_type == 'xy_cos':
            losses['rot_loss'] = (losses['rotx_loss'] + losses['roty_loss'] / 2)
        
        losses['loss'] = losses['pos_loss'] + losses['rot_loss']
        return losses
