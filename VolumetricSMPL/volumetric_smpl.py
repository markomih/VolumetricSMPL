import copy

import smplx
import torch
import trimesh
import numpy as np
import torch.nn.functional as F

from skimage import measure
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.point_mesh_distance import point_face_distance

from .modules import ResnetPointnet, ImplicitNet

JOINT_NAMES = [ 'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5',  'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1']
class Partitioner(torch.nn.Module):
    MERGE_BODY_PARTS = {
        'smpl': sorted([
            15,  # face into neck
            10,  # left toes into left foot
            11,  # right toes into right foot
            3,  # mid stomach into lower stomach
            13, 14,  # left and right shoulder blade
            22, 23,  # left and right hand into wrists
        ], reverse=True),

        'smplx': sorted([
            15,  # face into neck
            10,  # left toes into left foot
            11,  # right toes into right foot
            3,  # mid stomach into lower stomach
            13, 14,  # left and right shoulder blade
            9,  # upper body with upper stomach
        ], reverse=True),

        'mano': sorted([
            3,6,9,12,15, # finger tips
            2,5,8,11,14, # middle finger parts

        ], reverse=True),
    }
    SELFPEN_DISABLE_PARTS = {  # disable close connections for self-penetration
        'smpl': [
            (1, 2),  # L_Hip - R_Hip

            (1, 3),  # L_Hip - Spine1
            (2, 3),  # R_Hip - Spine1

            (3, 9),  # Spine1 - Spine3

            (9, 15),  # Spine3 - Head 
            (9, 16),  # Spine3 - L_Shoulder 
            (9, 17),  # Spine3 - R_Shoulder 

            (12, 6),  # Neck - Spline2
            (12, 16),  # Neck - L_Shoulder
            (12, 17),  # Neck - R_Shoulder

            (13, 6),  # L_Collar - Spine2
            (13, 12),  # L_Collar - Neck
            (14, 6),  # R_Collar - Spine2
            (14, 12),  # R_Collar - neck

            (0, 6),  # pelvis and spine2
            (0, 9),  # pelvis and spine3
            (0, 16),  # pelvis and L_Shoulder 
            (0, 17),  # pelvis and R_Shoulder 

            (6, 16),  # Spine2 - L_Shoulder
            (6, 17),  # Spine2 - R_Shoulder
        ],
        'smplx': [
            (1, 2),  # L_Hip - R_Hip

            (1, 3),  # L_Hip - Spine1
            (2, 3),  # R_Hip - Spine1

            (3, 9),  # Spine1 - Spine3

            (9, 15),  # Spine3 - Head
            (9, 16),  # Spine3 - L_Shoulder
            (9, 17),  # Spine3 - R_Shoulder

            (12, 6),  # Neck - Spline2
            (12, 16),  # Neck - L_Shoulder
            (12, 17),  # Neck - R_Shoulder

            (13, 6),  # L_Collar - Spine2
            (13, 12),  # L_Collar - Neck
            (14, 6),  # R_Collar - Spine2
            (14, 12),  # R_Collar - neck

            (0, 6),  # pelvis and spine2
            (0, 9),  # pelvis and spine3
            (0, 16),  # pelvis and L_Shoulder
            (0, 17),  # pelvis and R_Shoulder

            (6, 16),  # Spine2 - L_Shoulder
            (6, 17),  # Spine2 - R_Shoulder
        ],
        'mano': [
        ],
    }

    def __init__(self, parametric_body: smplx.SMPL):
        super().__init__()

        self.model_type = self.get_model_type(parametric_body)
        self.merge_body_parts = self.MERGE_BODY_PARTS[self.model_type]
        self.selfpen_disable_parts = self.SELFPEN_DISABLE_PARTS[self.model_type]

        (tight_face_tensor, extended_face_tensor), tight_vert_selector = self.partition_body(parametric_body)

        self.num_parts = len(tight_vert_selector)
        joint_mapper = torch.ones(self.num_parts+len(self.merge_body_parts), dtype=torch.bool)
        joint_mapper[self.merge_body_parts] = False

        # register variables
        self.faces = parametric_body.faces.copy()
        self.register_buffer('parents', parametric_body.parents.detach().clone().long())
        self.register_buffer('joint_mapper', joint_mapper)
        self.register_buffer('tight_face_tensor', tight_face_tensor)
        self.register_buffer('extended_face_tensor', extended_face_tensor)
        self.register_buffer('tight_vert_selector', tight_vert_selector)

        # variables for resolving self-intersections
        if 'smpl' in self.model_type:
            selfpen_disable_mat = self.get_selfpen_disable_mat(joint_mapper, 24)
            self.register_buffer('selfpen_disable_mat', selfpen_disable_mat, persistent=False)

    @torch.no_grad()
    def get_selfpen_disable_mat(self, joint_mapper, n_parts=24):
        # smpl_body = np.load(bm_path)
        # kintree_table = smpl_body['kintree_table'][0].astype(np.int32)[:n_parts]
        kintree_table = self.parents.cpu().numpy()[:n_parts]
        mat = np.ones((n_parts, n_parts), dtype=np.float32)
        for b_part in range(n_parts):
            # disable offsprings
            children_ind = [i for i in range(n_parts) if kintree_table[i] == b_part]
            mat[b_part, children_ind] = 0

            # disable parent nodes
            for ch in children_ind:
                parent_id = kintree_table[ch]
                if parent_id != -1:
                    mat[ch, parent_id] = 0

        for elem in self.selfpen_disable_parts:
            mat[elem[0], elem[1]] = 0
            mat[elem[1], elem[0]] = 0
        
        mat = torch.from_numpy(mat).to(device=joint_mapper.device)
        jm_inds = torch.where(joint_mapper)
        mat = mat[jm_inds].T[jm_inds].T

        return mat.bool()

    def check_bbox_penetrations(self, bb_min, bb_max, abs_transforms, EPS=1e-4):
        """
        Args:
            bb_min (torch.tensor): K,3
            bb_max (torch.tensor): K,3
            abs_transforms (torch.tensor): K,4,4
            disable_grad_mat (torch.tensor): K,K
        Returns:
        """
        K = bb_min.shape[0]
        m = bb_min.view(K, 3, 1)
        x = bb_max.view(K, 3, 1)
        corners = torch.cat([
            torch.cat((m[:, 0], m[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((m[:, 0], m[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((m[:, 0], x[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((m[:, 0], x[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], m[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], m[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], x[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], x[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
        ], dim=-2)  # K,8,3

        hcorners = torch.ones((K * 8, 4), dtype=bb_min.dtype, device=bb_min.device)
        hcorners[:, :3] = corners.view(-1, 3)
        hposed_corners = torch.bmm(
            abs_transforms.view(K, 1, 4, 4).repeat(1, 8, 1, 1).view(K * 8, 4, 4),
            hcorners.view(K * 8, 4, 1)
        ).view(K * 8, 4)

        # project posed corners to each other's space
        back_trans = torch.inverse(abs_transforms)

        proj_corners = torch.bmm(
            back_trans.view(K, 1, 4, 4).repeat(1, K * 8, 1, 1).view(K * K * 8, 4, 4),  # K*K*8,4,4
            hposed_corners.view(1, K * 8, 4).repeat(K, 1, 1).view(K * K * 8, 4, 1)  # K*K*8,4,1
        ).view(K, K, 8, 4)[..., :3]  # K, K, 8, 3

        # check which corner is inside of which box
        bb_min = bb_min.view(K, 1, 1, 3).repeat(1, K, 8, 1)  # (K, K, 8, 3)
        bb_max = bb_max.view(K, 1, 1, 3).repeat(1, K, 8, 1)  # (K, K, 8, 3)

        inside = ((proj_corners > bb_min + EPS) & (proj_corners < bb_max - EPS)).all(dim=-1)  # K, K, 8
        inside = inside.any(dim=-1)  # at least one corner needs to be inside
        # print('Prev collisions', inside.sum())
        inside = inside & self.selfpen_disable_mat

        # remove duplicates
        inside = inside | inside.t()  # symmetric table
        inside = torch.tril(inside)  # lower triangular part
        # print('After collisions', inside.sum())

        # assert torch.allclose(inside, inside.t())  # ensure this check is symmetric
        inds = torch.stack(torch.where(inside)).sort(dim=0)[0].t()
        # print(inds.cpu().numpy())
        return hposed_corners.view(K, 8, 4)[..., :3], inds

    @staticmethod
    def sample_collision_candidates(bb_min, bb_max, abs_transforms, collided_inds, n_points_uniform=300):
        """
        Args:
            bb_min (torch.tensor): K,3
            bb_max (torch.tensor): K,3
            abs_transforms (torch.tensor): K,4,4
            collided_inds (torch.tensor): N,2
            n_points_uniform (int):
        Returns:
        """
        def _check_inside_bbox(points, bwd_trans, box_min, box_max):  # (N,P,3), (N,4,4), (N,3)
            can_points = (bwd_trans.reshape(-1, 1, 4, 4).expand(-1, points.size(1), -1, -1) @ F.pad(points, [0, 1], "constant", 1.0).unsqueeze(-1))[:, :, :3, 0]
            p_in = ((can_points > box_min.unsqueeze(1)) & (can_points < box_max.unsqueeze(1))).all(dim=-1)
            return p_in  # (N, P)

        bb_min, bb_max = bb_min.reshape(-1, 3), bb_max.reshape(-1, 3)
        collided_inds = collided_inds.reshape(-1)  # (N*2,)

        c_bb_min = bb_min[collided_inds]  # (N*2,3)
        c_bb_max = bb_max[collided_inds]  # (N*2,3)
        bbox_size = (c_bb_max - c_bb_min)  # (N*2,3)
        uniform_points = torch.rand((bbox_size.size(0), n_points_uniform, 3), device=bb_min.device)  # (N*2,P,3)
        uniform_points = uniform_points * bbox_size.unsqueeze(1) + c_bb_min.unsqueeze(1)  # (N*2,P,3)

        # project points to the posed space
        fwd_trans = abs_transforms[collided_inds]  # (N*2, 4, 4)
        uniform_points = torch.matmul(
            fwd_trans.reshape(-1, 2, 1, 4, 4).expand(-1, -1, n_points_uniform, -1, -1),  # (N,2,P,4,4)
            F.pad(uniform_points, [0, 1], "constant", 1.0).reshape(-1, 2, n_points_uniform, 4, 1)  # (N,2, P, 4, 1)
        )[..., :3, 0]  # (N, 2, P, 3)

        # keep only samples that are inside both colliding boxes
        bwd_trans = torch.inverse(abs_transforms)[collided_inds].view(-1, 2, 4, 4)  # (N, 2, 4, 4)
        c_bb_min, c_bb_max = c_bb_min.view(-1, 2, 3), c_bb_max.view(-1, 2, 3)  # (N, 2, 3)

        in1 = _check_inside_bbox(uniform_points[:, 0, ...], bwd_trans[:, 1, ...], c_bb_min[:, 1, ...], c_bb_max[:, 1, ...])
        in2 = _check_inside_bbox(uniform_points[:, 1, ...], bwd_trans[:, 0, ...], c_bb_min[:, 0, ...], c_bb_max[:, 0, ...])

        collision_candidates1 = uniform_points[:, 0, ...][in1]
        collision_candidates2 = uniform_points[:, 1, ...][in2]
        collision_candidates = torch.cat((collision_candidates1, collision_candidates2), dim=0)
        return collision_candidates

    @staticmethod
    def get_model_type(parametric_body):
        if type(parametric_body) == smplx.MANO:
            model_type = 'mano'
        elif type(parametric_body) == smplx.FLAME:
            model_type = 'flame'
        elif type(parametric_body) == smplx.SMPLX:
            model_type = 'smplx'
        elif type(parametric_body) == smplx.SMPLH:
            model_type = 'smplh'
        elif type(parametric_body) == smplx.SMPL:
            model_type = 'smpl'
        else:
            raise NotImplementedError('The body type is not supported.')
        return model_type

    def get_bbox_bounds(self, vertices, bone_trans):
        """
        Args:
            vertices (torch.tensor): (B,V,3)
            bone_trans (torch.tensor): (B,K,4,4)
        Returns:
            tuple:
                bbox_min (torch.tensor): BxKx1x3
                bbox_max (torch.tensor): BxKx1x3
        """
        B, K = bone_trans.shape[0], bone_trans.shape[1]
        max_select_dim = self.tight_vert_selector.shape[1]

        part_vertices = torch.index_select(vertices, 1, self.tight_vert_selector.view(-1))  # (B, K*max_select_dim, 3)
        part_vertices = part_vertices.view(B, K, max_select_dim, 3)

        bone_trans = bone_trans.unsqueeze(2).expand(-1, -1, max_select_dim, -1, -1)

        local_part_vertices = (bone_trans @ F.pad(part_vertices, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]
        bbox_min = local_part_vertices.min(dim=-2, keepdim=True).values  # (B, K, 1, 3)
        bbox_max = local_part_vertices.max(dim=-2, keepdim=True).values  # (B, K, 1, 3)

        return bbox_min, bbox_max

    @torch.no_grad()
    def partition_body(self, parametric_body):
        # prepare data
        faces = self.tensor_to_numpy(parametric_body.faces).copy()
        v_template = self.tensor_to_numpy(parametric_body.v_template).copy()
        lbs_weights = self.tensor_to_numpy(parametric_body.lbs_weights).copy()
        parents = self.tensor_to_numpy(parametric_body.parents.long()).copy()

        max_part = lbs_weights.argmax(axis=1)
        can_mesh = trimesh.Trimesh(v_template, faces, process=False)
        kintree_table = parents

        # merge parts
        if self.model_type == 'smplx':
            # merge jaw into face and hand articulation
            max_part[(max_part == 22)] = kintree_table[22]  # jaw
            max_part[(max_part >= 25) & (max_part < 40)] = kintree_table[25 + 3]  # merge left hand
            max_part[(max_part >= 40) & (max_part < 55)] = kintree_table[40 + 3]  # merge right hand
            max_part[max_part >= 22] = 100  # invalid labels for eyes
            n_parts = 22
        elif self.model_type == 'smpl':
            n_parts = 24
        else:
            n_parts = kintree_table.shape[0]

        kintree_table = kintree_table[:n_parts]
        valid_labels = list(range(n_parts))
        
        # merge small body parts
        for ind in self.merge_body_parts:
            # relabel vertices
            max_part[(max_part == ind)] = kintree_table[ind]
            valid_labels[ind] = kintree_table[ind]

            # update kintable; all children take a grandparent's id
            kintree_table[kintree_table == ind] = kintree_table[ind]

        # # convert vertex labels
        # face labels
        edges = can_mesh.edges_unique
        faces_unique_edges = can_mesh.faces_unique_edges  # F,3
        lab_list = []
        for f_ind in range(can_mesh.faces.shape[0]):
            vert_id = edges[faces_unique_edges[f_ind]].reshape(-1)  # (6,)
            vert_id = np.unique(vert_id)  # (3,)
            vert_sk_lab = max_part[vert_id]  # (3,)
            lab_list.append(np.bincount(vert_sk_lab).argmax())

        face_labels = np.array(lab_list)

        tight_face_list, extended_face_list = [], []
        tight_face_mask_list, extended_face_mask_list = [], []
        tight_vertex_mask_list, extended_vertex_mask_list = [], []
        for b_part in range(n_parts):
            if b_part in self.merge_body_parts:
                continue

            b_label = valid_labels[b_part]
            tight_inds, extended_inds = face_labels == b_label, face_labels == b_label
            parent_ind = kintree_table[b_part]
            ch_inds = [i for i in range(n_parts) if kintree_table[i] == b_part]
            ch_label = [valid_labels[ch] for ch in ch_inds]

            if parent_ind != -1:
                parent_label = valid_labels[parent_ind]
                extended_inds = extended_inds | (face_labels == parent_label)
            for ch_lab in ch_label:
                extended_inds = extended_inds | (face_labels == ch_lab)

            tight_face_list.append(can_mesh.faces[tight_inds])
            tight_face_mask_list.append(tight_inds)
            extended_face_list.append(can_mesh.faces[extended_inds])
            extended_face_mask_list.append(extended_inds)

            tight_vertex_inds, extended_vertex_inds = max_part == b_label, max_part == b_label
            if parent_ind != -1:
                extended_vertex_inds = extended_vertex_inds | (max_part == parent_label)
            for ch_lab in ch_label:
                extended_vertex_inds = extended_vertex_inds | (max_part == ch_lab)

            tight_vertex_mask_list.append(tight_vertex_inds)
            extended_vertex_mask_list.append(extended_vertex_inds)

        tight_face_tensor, extended_face_tensor = [], []
        tight_max_size = max([x.shape[0] for x in tight_face_list])
        extended_max_size = max([x.shape[0] for x in extended_face_list])
        for x in tight_face_list:
            padded_face_tensor = np.full((tight_max_size, 3), fill_value=-1, dtype=np.int32)
            padded_face_tensor[:x.shape[0], :] = x
            tight_face_tensor.append(torch.from_numpy(padded_face_tensor).unsqueeze(0))

        for x in extended_face_list:
            padded_face_tensor = np.full((extended_max_size, 3), fill_value=-1, dtype=np.int32)
            padded_face_tensor[:x.shape[0], :] = x
            extended_face_tensor.append(torch.from_numpy(padded_face_tensor).unsqueeze(0))

        tight_face_tensor = torch.cat(tight_face_tensor, dim=0)
        extended_face_tensor = torch.cat(extended_face_tensor, dim=0)
        tight_vert_selector = [
            torch.where(torch.from_numpy(max_part == valid_labels[i]))[0]
            for i in range(n_parts) if i not in self.merge_body_parts
        ]
        # tight_vert_selector = [torch.from_numpy(vert_ids) for vert_ids in max_part]
        max_elem = max([v_sec.shape[0] for v_sec in tight_vert_selector])
        for b_part in range(len(tight_vert_selector)):
            vert_sec = torch.full((max_elem,), dtype=tight_vert_selector[b_part].dtype,
                                fill_value=tight_vert_selector[b_part][0].item())
            vert_sec[:tight_vert_selector[b_part].shape[0]] = tight_vert_selector[b_part]
            tight_vert_selector[b_part] = vert_sec

        tight_vert_selector = torch.stack(tight_vert_selector)
        return (tight_face_tensor, extended_face_tensor), tight_vert_selector

    @staticmethod
    def _sample_mesh_points(posed_vert, face_tensor, n_samples, return_normals=False):
        B, V, _ = posed_vert.shape
        n_parts = face_tensor.shape[0]
        meshes = Meshes(  # (B*n_parts,V,3)
            verts=posed_vert.unsqueeze(1).expand(-1, n_parts, -1, -1).contiguous().view(B * n_parts, -1, 3),
            faces=face_tensor.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * n_parts, -1, 3))
        samp_points = sample_points_from_meshes(meshes, n_samples, return_normals)
        if return_normals:
            return samp_points[0].view(B, n_parts, n_samples, 3), samp_points[1].view(B, n_parts, n_samples, 3)
        return samp_points.view(B, n_parts, n_samples, 3)
    
    def sample_mesh_points(self, posed_vert, n_samples, return_normals=False):
        n_tight_samples = n_samples // 2
        n_ext_samples = n_samples - n_tight_samples
        central_points = self._sample_mesh_points(posed_vert, self.tight_face_tensor, n_tight_samples, return_normals)
        uniform_points = self._sample_mesh_points(posed_vert, self.extended_face_tensor, n_ext_samples, return_normals)
        if return_normals:
            return torch.cat((central_points[0], uniform_points[0]), dim=-2), torch.cat((central_points[1], uniform_points[1]), dim=-2)
        return torch.cat((central_points, uniform_points), dim=-2)

    def compute_abs_transformations(self, full_pose, posed_joints):
        # full_pose: B, K*3 or B,K,3
        # posed_joints: B, K*3 or B,K,3
        B = full_pose.shape[0]
        full_pose = full_pose.reshape(B, -1, 3)
        posed_joints = posed_joints.reshape(B, -1, 3)
        K = full_pose.shape[1]
        mK = self.joint_mapper.shape[0]
        # posed_joints = posed_joints[:, :K]  # remove extra joints (landmarks for smplx)
        
        # torchgeometry.angle_axis_to_rotation_matrix(full_pose.view(-1, 3))[:, :3, :3]
        rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view(B, K, 3, 3)  # B*K,3,3

        # fwd lbs to estimate absolute transformation matrices
        parents = self.parents.long()
        transform_chain = [rot_mats[:, 0]]
        for i in range(1, parents.shape[0]):
            if i == mK:
                break
            transform_chain.append(transform_chain[parents[i]] @ rot_mats[:, i])

        transforms = torch.stack(transform_chain, dim=1)
        abs_trans = torch.cat([
            F.pad(transforms.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(posed_joints[:, :mK].reshape(-1, 3, 1), [0, 0, 0, 1], value=1)
        ], dim=2).reshape(B, mK, 4, 4)

        # remap joints
        abs_trans = abs_trans.transpose(0, 1)[self.joint_mapper].transpose(0, 1)
        # TODO: get pose feature as abs_trans - abs_trans[self.joint_mapper]
        return abs_trans

    @staticmethod
    def tensor_to_numpy(tensor_vec):
        if torch.is_tensor(tensor_vec):
            return tensor_vec.detach().cpu().numpy()
        return tensor_vec

def point_mesh_distance(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        the closest distance from a point to the mesh.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return torch.sqrt(point_to_face)

class BasicBodyModel(torch.nn.Module):

    def __init__(self, parametric_body: smplx.SMPL, body_cfg={}) -> None:
        super().__init__()
        self._init(parametric_body, body_cfg)

    def _init(self, parametric_body: smplx.SMPL, body_cfg={}) -> None:
        self.body_cfg = body_cfg

        # hyperparameters
        self.n_samples = self.body_cfg.get('n_samples', 1000)
        self.bbox_padding = 1.125  # bounding box size

        # create differentiable modules
        self.partitioner = Partitioner(parametric_body)
        self.encoder = self.create_encoder()
        self.query_encoder = ImplicitNet(
            d_in=3+1+self.partitioner.num_parts+self.encoder.out_dim,  # 3 + 128 (pn code) + 1 for mask + 24 for one hot
            d_out=self.encoder.out_dim,
            dims=[ 256, 256, 256 ],
            skip_in=[2],
            geometric_init=False,
        )
        self.decoder = self.create_decoder()
        self.level_set = 0.5
        self.register_buffer('k_one_hot', torch.eye(self.partitioner.num_parts, dtype=torch.float32))
        self.out_act = torch.sigmoid
        self.impl_code = None
        self.model_type = self.partitioner.model_type

    def create_encoder(self):
        resnet_kwargs = self.body_cfg.get('resnet_kwargs', dict(out_dim=128, hidden_dim=128, dim=3, use_block2=False))
        encoder = ResnetPointnet(**resnet_kwargs)
        return encoder

    def create_decoder(self):
        decoder = ImplicitNet(
            d_in=3+self.query_encoder.d_out,
            d_out=1,
            dims=[ 256, 256, 256, 256, 256, 256 ],
            skip_in=[3],
        )
        return decoder

    def get_bbox_bounds(self, vertices, bone_trans):
        bbox_min, bbox_max = self.partitioner.get_bbox_bounds(vertices, bone_trans)  # (B, K, 1, 3)
        return bbox_min, bbox_max
    
    def get_tight_face_tensor(self):
        return self.partitioner.tight_face_tensor.contiguous()

    def compute_bone_trans(self, full_pose, joints):
        abs_transforms = self.partitioner.compute_abs_transformations(full_pose, joints)
        bone_trans = torch.inverse(abs_transforms)
        return bone_trans

    def encode_body(self, smpl_output):
        full_pose = smpl_output.full_pose
        joints = smpl_output.joints
        vertices = smpl_output.vertices

        # estimate absolute transformation matrices
        bone_trans = self.compute_bone_trans(full_pose, joints)
        bbox_min, bbox_max = self.get_bbox_bounds(vertices, bone_trans)  # (B, K, 1, 3)

        # estimate absolute transformation matrices
        sampled_points = self.partitioner.sample_mesh_points(vertices, self.n_samples)
        B, K, n_points, _ = sampled_points.shape

        # canonicalize sampled points [B,K,T,3]
        local_sampled_points = (bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, n_points, -1, -1) @ F.pad(sampled_points, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]  # B,K,T,3

        # encode local point clouds
        latent_code = self.encoder(local_sampled_points.reshape(B*K, -1, 3)).view(B, K, -1)  # (B,K,f_len)

        # store data
        self.impl_code = dict(
            full_pose=full_pose,
            joints=joints,
            bone_trans=bone_trans,
            latent_code=latent_code,
            bbox_min=bbox_min, bbox_max=bbox_max,
            bbox_size=torch.abs(bbox_max - bbox_min)*self.bbox_padding,
            bbox_center=(bbox_min + bbox_max)*0.5  # (B, K, 1, 3),
        )

    def _attach_impl_code(self, smpl_output=None):
        if smpl_output is None:
            return

        # recalculate encodings if input has changed
        if self.impl_code is None:
            assert smpl_output is not None
            self.encode_body(smpl_output)
        elif smpl_output.full_pose.shape != self.impl_code['full_pose'].shape or\
            not torch.allclose(smpl_output.full_pose, self.impl_code['full_pose']) or\
            not torch.allclose(smpl_output.joints, self.impl_code['joints']):
            self.encode_body(smpl_output)

    def detach_cache(self):
        self.impl_code = None

    def query_occupancy(self, points, smpl_output=None, ret_intermediate=False):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        return self.query(points, smpl_output, ret_intermediate)

    def query(self, points, smpl_output=None, ret_intermediate=False):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        # attach variables
        self._attach_impl_code(smpl_output)

        latent_code = self.impl_code['latent_code']
        bone_trans = self.impl_code['bone_trans']
        bbox_size = self.impl_code['bbox_size']
        bbox_center = self.impl_code['bbox_center']

        B, K = latent_code.shape[:2]
        T = points.shape[1]

        # canonicalize query points
        local_queries = (bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, T, -1, -1) @ F.pad(points, [0, 1], "constant", 1.0).reshape(B, 1, T, 4, 1).expand(-1, K, -1, -1, -1))[..., :3, 0]
        inside_bbox = ((local_queries - bbox_center).abs() < (bbox_size * 0.5)).all(dim=-1).int()  # (B, K, T)

        # blend features and local queries
        z_code = torch.cat((
            local_queries,  # B,K,T,3
            inside_bbox.unsqueeze(-1), # B,K,T,1
            self.k_one_hot.reshape(1, K, 1, K).expand(B, -1, T, -1),  # (B,K,T,K)
            latent_code.unsqueeze(-2).expand(-1, -1, T, -1),  # (B,K,T,f_len)
        ), dim=-1).reshape(B*K, T, -1)
        z_code = self.query_encoder(z_code)  # (B*K,T,d_len)
        z_code = torch.cat((local_queries.reshape(B * K, -1, 3), z_code), dim=-1)

        # query occupancy
        part_occupancy = self.decoder(z_code).reshape(B, K, -1)
        part_occupancy = self.out_act(-part_occupancy)
        part_occupancy = part_occupancy*inside_bbox  # (B, K, T)
        occupancy = part_occupancy.max(dim=1).values  # (B, T)

        if ret_intermediate:
            return occupancy, {'part_occupancy': part_occupancy, 'all_out': ~inside_bbox.any(1)}

        return occupancy

    # def collision_loss(self, point_cloud, smpl_output, ret_collision_mask=False):
    #     """
    #     Args:
    #         point_cloud (list or torch.tensor): B, N, 3
    #     """
    #     occupancy = self.query(point_cloud, smpl_output)
    #     loss = torch.relu(occupancy - self.level_set)  # B,N
    #     if ret_collision_mask is not None:
    #         inds = loss > 0
    #         return loss.sum(-1), inds
    #     return loss.sum(-1)  # B, N, 3

    @staticmethod
    def batchify_smpl_output(smpl_output):
        b_smpl_output_list = []
        batch_size = smpl_output.vertices.shape[0]
        for b_ind in range(batch_size):
            b_smpl_output_list.append(copy.copy(smpl_output))
            for key in b_smpl_output_list[-1].keys():
                val = getattr(smpl_output, key)
                if torch.is_tensor(val):
                    val = val[b_ind:b_ind+1].clone()
                setattr(b_smpl_output_list[-1], key, val)
        return b_smpl_output_list

    @torch.no_grad()
    def get_gt_sdf(self, smpl_output, points, gt_occ):
        B, T = points.shape[:2]
        gt_udf = point_mesh_distance(
            Meshes(verts=smpl_output.vertices, faces=torch.from_numpy(self.partitioner.faces[None].astype(np.int64)).to(points.device).expand(B, -1, -1)),
            Pointclouds(points)).reshape(B, T)
        gt_sdf = torch.where(gt_occ.bool(), -gt_udf, gt_udf)
        return gt_sdf

    @torch.no_grad()
    def extract_mesh(self, smpl_output, grid_res=64, max_queries=100000, use_mise=False, mise_resolution0=32, mise_depth=3):
        scale = 1.1  # padding

        act = lambda x: torch.log(x/(1-x+1e-6)+1e-6)  # revert sigmoid
        level = 0.0
        occ_list = []

        verts = smpl_output.vertices
        B = verts.shape[0]
        device = verts.device
        part_colors = self.get_part_colors()
        b_smpl_output_list = self.batchify_smpl_output(smpl_output)

        b_min, b_max = verts.min(dim=1).values, verts.max(dim=1).values  # B,3
        gt_center = ((b_min + b_max)*0.5).cpu()  # B,3
        gt_scale = (b_max - b_min).max(dim=-1, keepdim=True).values.cpu()  # (B,1)
        gt_scale_gpu, gt_center_gpu = gt_scale.to(device), gt_center.to(device)

        # query grid
        if use_mise:
            from leap.tools.libmise import MISE
            value_grid_list = []
            for b_ind in range(B):
                mesh_extractor = MISE(mise_resolution0, mise_depth, level)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    grid3d = torch.FloatTensor(points).to(device)
                    grid3d = scale*(grid3d/mesh_extractor.resolution - 0.5).reshape(1, -1, 3)  # [-0.5, 0.5]*scale
                    grid3d = grid3d*gt_scale_gpu[b_ind].reshape(1, 1, 1) + gt_center_gpu[b_ind].reshape(1, 1, 3)

                    # check occupancy for sampled points
                    occ_hats = []
                    for pts in torch.split(grid3d, max_queries, dim=1):
                        occ_hats.append(act(self.query_occupancy(pts.to(device=device), b_smpl_output_list[b_ind])).cpu().squeeze(0))  # N
                    values = torch.cat(occ_hats, dim=0).numpy().astype(np.float64)

                    # sample points again
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                value_grid_list.append(mesh_extractor.to_dense())
            value_grid = np.stack(value_grid_list)
            grid_res = mesh_extractor.resolution
        else:
            grid3d = self.create_meshgrid3d(grid_res, grid_res, grid_res)  # range [0,G]
            grid3d = scale*(grid3d/grid_res - 0.5).reshape(1, -1, 3)  # (1,D*H*W,3) in range [-0.5, +0.5]*scale
            for grid_queries in torch.split(grid3d, max_queries//B, dim=1):
                pts = grid_queries.expand(B, -1, -1).to(device)*gt_scale_gpu.unsqueeze(1) + gt_center_gpu.unsqueeze(1)
                occ_list.append(act(self.query(pts, smpl_output)).cpu())  # B,N
            value_grid = torch.cat(occ_list, dim=1).reshape(B, grid_res, grid_res, grid_res).numpy()  # B,D,H,W

        # extract meshes
        mesh_list = []
        for b_ind in range(B):
            _lvl = level
            if value_grid[b_ind].max() < _lvl:
                _lvl = value_grid[b_ind].max() - 1e-6
            verts, faces, normals, values = measure.marching_cubes(volume=value_grid[b_ind], gradient_direction='ascent', level=_lvl)

            # vertices to world space
            verts = scale*(verts/(grid_res-1) - 0.5)
            verts = verts*gt_scale[b_ind].item() + gt_center[b_ind].cpu().numpy()
            
            # color meshes
            vertex_colors = self.color_points(torch.from_numpy(verts).reshape(1, -1, 3).to(device), b_smpl_output_list[b_ind], max_queries)[0]
            mesh_list.append(trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors))

        return mesh_list

    @torch.no_grad()
    def color_points(self, pts, smpl_output=None, max_queries=100000):
        part_colors = self.get_part_colors()
        K = self.partitioner.num_parts
        B = pts.shape[0]
        part_colors = part_colors[:K+1]
        part_colors[-1, :] = 0  # set bg to black

        pts_colors = []
        for _v in torch.split(pts, max_queries//B, dim=1):
            part_pred = self.query(_v, smpl_output, ret_intermediate=True)[1]
            label = part_pred['part_occupancy'].argmax(dim=1)  # B,K,V -> B,V
            label[part_pred['all_out']] = K  # all mlps say outside
            inds = label.reshape(-1).cpu().numpy()
            pts_colors.append(part_colors[inds].reshape((B, -1, 3)))

        pts_colors = np.concatenate(pts_colors, 1)
        return pts_colors

    @staticmethod
    def create_meshgrid3d(
        depth: int,
        height: int,
        width: int,
        device=torch.device('cpu'),
        dtype=torch.float32,
    ) -> torch.Tensor:
        """ Generate a coordinate grid in range [-0.5, 0.5].

        Args:
            depth (int): grid dim
            height (int): grid dim
            width (int): grid dim
        Return:
            grid tensor with shape :math:`(1, D, H, W, 3)`.
        """
        xs = torch.linspace(0, width, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height, height, device=device, dtype=dtype)
        zs = torch.linspace(0, depth, depth, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid([xs, ys, zs]), dim=-1).unsqueeze(0)  # 1xDxHxWx3

    @staticmethod
    def compute_iou(occ1, occ2, level=0.5):
        """ Computes the Intersection over Union (IoU) value for two sets of occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values  (B, N)
            occ2 (tensor): second set of occupancy values  (B, N)
            level (float): threshold

        Returns:
            iou (tensor): mean IoU (scalar)
        """
        if occ1.dtype != torch.bool:
            occ1 = (occ1 >= level)
            occ2 = (occ2 >= level)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).float().sum(axis=-1)

        iou = (area_intersect / area_union.clamp_min(1))

        return iou.mean()

    @staticmethod
    def get_part_colors():
        return np.array([
            [ 8.94117647e-01,  1.01960784e-01,  1.09803922e-01],
            [ 2.15686275e-01,  4.94117647e-01,  7.21568627e-01],
            [ 3.01960784e-01,  6.86274510e-01,  2.90196078e-01],
            [ 5.96078431e-01,  3.05882353e-01,  6.39215686e-01],
            [ 1.00000000e+00,  4.98039216e-01,  3.58602037e-16],
            [ 1.00000000e+00,  1.00000000e+00,  2.00000000e-01],
            [ 6.50980392e-01,  3.37254902e-01,  1.56862745e-01],
            [ 9.68627451e-01,  5.05882353e-01,  7.49019608e-01],
            [ 6.00000000e-01,  6.00000000e-01,  6.00000000e-01],
            [ 1.00000000e+00,  9.17647059e-01,  8.47058824e-01],
            [ 4.94117647e-01, -3.58602037e-16,  1.84313725e-01],
            [ 7.92156863e-01,  7.29411765e-01,  3.72549020e-01],
            [ 5.25490196e-01,  4.78431373e-01,  3.58602037e-16],
            [ 9.29411765e-01,  6.50980392e-01,  5.76470588e-01],
            [ 5.05882353e-01,  4.03921569e-01,  4.15686275e-01],
            [ 7.80392157e-01,  3.56862745e-01,  4.43137255e-01],
            [ 6.86274510e-01,  5.37254902e-01,  3.52941176e-01],
            [ 6.27450980e-01,  8.23529412e-02,  0.00000000e+00],
            [ 1.00000000e+00,  4.35294118e-01,  3.80392157e-01],
            [ 7.37254902e-01, -7.17204074e-16,  2.82352941e-01],
            [ 1.00000000e+00,  9.09803922e-01,  5.52941176e-01],
            [ 1.00000000e+00,  7.84313725e-02,  4.15686275e-01],
            [ 7.72549020e-01,  7.52941176e-01,  6.66666667e-01],
            [ 5.68627451e-01,  2.62745098e-01,  2.98039216e-01],
            [ 0,  0,  0],
        ])

class VolumetricSMPL(BasicBodyModel): #  volSMPL model
    def _init(self, parametric_body: smplx.SMPL, body_cfg={}) -> None:
        self.body_cfg = body_cfg

        # hyperparameters
        self.n_samples = self.body_cfg.get('n_samples', 1000)
        self.bbox_padding = 1.125  # bounding box size

        self.sdf_regularization = self.body_cfg.get('sdf_regularization', False)
        self.igr_normalize_part_loss = self.body_cfg.get('igr_normalize_part_loss', False)
        self.cond_decoder = self.body_cfg.get('cond_decoder', True)
        self.part_weights = self.body_cfg.get('part_weights', True)
        self.part_rank = self.body_cfg.get('part_rank', 10)

        self.rf_kwargs = self.body_cfg.get('rf_kwargs', dict(rank=10))
        self.decoder_name = self.body_cfg.get('decoder_name', 'ImplicitNet')
        self.decoder_cls = globals()[self.decoder_name]
        self.decoder_dims = self.body_cfg.get('decoder_dims', [ 256, 256, 256, 256, 256, 256 ])
        self.decoder_skip_in = self.body_cfg.get('decoder_skip_in', [4])
        self.decoder_multires = self.body_cfg.get('decoder_multires', 0)

        self.loss_weights = self.body_cfg.get('loss_weights', {})

        # create differentiable modules
        self.partitioner = Partitioner(parametric_body)
        self.encoder = self.create_encoder()
        self.rf_kwargs['cond_dim'] = self.encoder.out_dim
        if self.part_weights is not None:
            self.rf_kwargs['ind_dim'] = self.partitioner.num_parts
            self.rf_kwargs['part_rank'] = self.part_rank
            self.register_buffer('part_ind', torch.arange(self.partitioner.num_parts)[None])
        self.decoder = self.create_decoder()
        self.level_set = 0.5
        self.impl_code = None
        self.model_type = self.partitioner.model_type
    
    def create_encoder(self):
        resnet_kwargs = self.body_cfg.get('resnet_kwargs', dict(out_dim=128, hidden_dim=128, dim=3, use_block2=False))
        encoder = ResnetPointnet(**resnet_kwargs)
        return encoder

    def create_decoder(self):
        decoder = self.decoder_cls(
            d_in=3+self.encoder.out_dim if self.cond_decoder else 3,
            d_out=2,
            dims=self.decoder_dims,
            skip_in=self.decoder_skip_in,
            multires=self.decoder_multires,
            rf_kwargs=self.rf_kwargs,
        )
        return decoder

    def _fwd_pass(self, local_queries, inside_bbox, only_part_occupancy=False):
        """
        Args:
            local_query (torch.tensor): B,K, T, 3
        Returns:
            local part sdf (torch.tensor): B*K, T
        """
        B,K,T,_ = local_queries.shape
        z_code = local_queries
        latent_code = self.impl_code['latent_code'] # B,K, f_len
        if self.cond_decoder:
            z_code = torch.cat((
                z_code,  # B,K,T,3
                latent_code.unsqueeze(-2).expand(-1, -1, local_queries.shape[-2], -1),  # (B,K,T,f_len)
            ), dim=-1)
        ind = None
        if self.part_weights:
            ind = self.part_ind.expand(B, -1).reshape(-1) # B*K

        part_pred = self.decoder(z_code.reshape(B*K, T, -1), cond=latent_code.view(B*K, -1), ind=ind).reshape(B, K, T, 2)
        part_occ, part_udf = part_pred.split(1, dim=-1)

        # fuse occupancy
        part_occupancy = torch.sigmoid(-part_occ).squeeze(-1)
        part_occupancy = part_occupancy*inside_bbox  # (B, K, T)
        if only_part_occupancy:
            return part_occupancy
        occupancy = part_occupancy.max(dim=1).values  # (B, T)

        # fuse udf
        part_udf = torch.abs(part_udf).squeeze(-1)
        # part_udf_fused = torch.where(inside_bbox, part_udf, torch.full_like(part_udf, float('inf'))) # propagate loss only for points inside bboxes
        # udf = part_udf_fused.min(1).values  # (B, T)
        # pred_sdf = torch.where(occupancy > 0.5, -udf, udf) # B,T

        part_sdf = torch.where(part_occupancy > 0.5, -part_udf, part_udf) # B,K,T

        return occupancy, part_sdf, part_udf

    @torch.jit.script
    def to_local(points, bone_trans, bbox_center, bbox_size):
        B, K = bbox_size.shape[:2]
        T = points.shape[1]
        local_queries = (bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, T, -1, -1) @ F.pad(points, [0, 1], "constant", 1.0).reshape(B, 1, T, 4, 1).expand(-1, K, -1, -1, -1))[..., :3, 0]
        inside_bbox = ((local_queries - bbox_center).abs() < (bbox_size * 0.5)).all(dim=-1)  # (B, K, T)
        return local_queries, inside_bbox

    @torch.no_grad()
    def _sample_part_udf(self, smpl_output, n_unif_samples=512):
        bone_trans = self.impl_code['bone_trans']
        fwd_bone_trans = torch.inverse(bone_trans)
        bbox_size = self.impl_code['bbox_size']
        bbox_center = self.impl_code['bbox_center']
        latent_code = self.impl_code['latent_code']
        B, K = latent_code.shape[:2]

        # sample local queries
        bb_min = (bbox_center - bbox_size*0.5)
        b_size = bbox_size - 1e-4  # small padding
        b_min =  (bbox_center - b_size*0.5)  # to account for padding
        unif_local_queries = b_min + torch.rand((B, K, n_unif_samples, 3), device=bb_min.device) * b_size

        unif_posed_queries = torch.matmul(
            fwd_bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, n_unif_samples, -1, -1),
            F.pad(unif_local_queries, [0, 1], "constant", 1.0).unsqueeze(-1)
        )[..., :3, 0]  # B,K,n_surf,3
        gt_udf_unif_local_queries = point_mesh_distance(Meshes(
            verts=smpl_output.vertices.unsqueeze(1).expand(-1, K, -1, -1).contiguous().view(B * K, -1, 3),
            faces=self.partitioner.extended_face_tensor.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * K, -1, 3)),
            Pointclouds(unif_posed_queries.reshape(B*K, -1, 3))
        ).reshape(B, K, -1) # B,K,T

        return unif_local_queries, gt_udf_unif_local_queries

    def query_training(self, points, gt_occ, smpl_output, **kwargs):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        # attach variables
        loss = {}
        self._attach_impl_code(smpl_output)
        
        local_queries, inside_bbox = self.to_local(points, self.impl_code['bone_trans'], self.impl_code['bbox_center'], self.impl_code['bbox_size'])

        pred_occ = self._fwd_pass(local_queries, inside_bbox)[0] # B,T

        # occ loss
        all_out = ~inside_bbox.any(1)
        pred_occ = pred_occ[~all_out]
        loss['mse_occ'] = F.mse_loss(pred_occ, gt_occ[~all_out])

        # local udf loss
        unif_local_pts, udf_unif_local_pts = self._sample_part_udf(smpl_output, n_unif_samples=512) # [B,K,T',3], [B,K,T']
        pred_unif_local_pts = self._fwd_pass(unif_local_pts, torch.ones_like(unif_local_pts[..., 0]).bool())[-1]
        loss['mse_udf'] = F.mse_loss(udf_unif_local_pts, pred_unif_local_pts)

        # fuse loss terms
        loss = {key: val*self.loss_weights.get(key, 1.0) for key, val in loss.items()}
        loss['total_loss'] = sum(val for key, val in loss.items())
        return loss

    @torch.jit.script
    def _get_analytic_part_sdf(local_queries, bbox_center, bbox_offset):
        _sh_q = (local_queries - bbox_center).abs() - bbox_offset   # use the tightest box to estimate sdf
        part_bbox_sdf = torch.norm(_sh_q.clamp_min(0.0), dim=-1) + _sh_q.max(-1).values.clamp_max(0.0) # (B, K, T)
        return part_bbox_sdf

    def query(self, points, smpl_output=None, ret_intermediate=False):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        return self.query_fast(points, smpl_output, ret_intermediate)
        # attach variables
        self._attach_impl_code(smpl_output)

        local_queries, inside_bbox = self.to_local(points, self.impl_code['bone_trans'], self.impl_code['bbox_center'], self.impl_code['bbox_size'])
        part_bbox_sdf = self._get_analytic_part_sdf(local_queries, self.impl_code['bbox_center'], self.impl_code['bbox_size']*(0.5/self.bbox_padding))
        
        part_sdf = self._fwd_pass(local_queries, inside_bbox)[1] # B,K,T
        # # query sdf
        fused_part_sdf = torch.where(inside_bbox, part_sdf, part_bbox_sdf)
        sdf = fused_part_sdf.min(dim=1).values

        if ret_intermediate:
            return sdf, {'part_sdf': fused_part_sdf}

        return sdf
    
    def query_fast(self, points, smpl_output=None, ret_intermediate=False):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        # attach variables
        self._attach_impl_code(smpl_output)

        local_queries, inside_bbox = self.to_local(points, self.impl_code['bone_trans'], self.impl_code['bbox_center'], self.impl_code['bbox_size'])
        part_bbox_sdf = self._get_analytic_part_sdf(local_queries, self.impl_code['bbox_center'], self.impl_code['bbox_size']*(0.5/self.bbox_padding))
        
        # filter out queries that don't need to be evaluated
        filter_pts = inside_bbox.view(-1, inside_bbox.shape[-1]).any(0) # B,K,T -> T
        _local_queries = local_queries[:, :, filter_pts, :] # B,K,T',3
        part_sdf = torch.empty_like(part_bbox_sdf)
        if filter_pts.any():
            part_sdf[:, :, filter_pts] = self._fwd_pass(_local_queries, inside_bbox[:, :, filter_pts])[1] # B,K,T
        # # query sdf
        fused_part_sdf = torch.where(inside_bbox, part_sdf, part_bbox_sdf)
        sdf = fused_part_sdf.min(dim=1).values
        if ret_intermediate:
            return sdf, {'part_sdf': fused_part_sdf}

        return sdf

    def collision_loss(self, point_cloud, smpl_output, ret_collision_mask=False):
        """
        Args:
            point_cloud (list or torch.tensor): B, N, 3
        """
        sdf_pred = self.query_fast(point_cloud, smpl_output)
        loss = torch.relu(-sdf_pred)  # B,N
        if ret_collision_mask is not None:
            inds = loss > 0
            return loss.sum(-1), inds
        return loss.sum(-1)  # B, N, 3

    def collision_loss_mean(self, point_cloud, smpl_output, ret_collision_mask=False):
        """
        Args:
            point_cloud (list or torch.tensor): B, N, 3
        """
        sdf_pred = self.query_fast(point_cloud, smpl_output)
        loss = torch.relu(-sdf_pred)  # B,N
        inds = loss > 0
        loss = loss.sum(-1)/(inds.float().sum(-1)+1e-6)
        if ret_collision_mask is not None:
            inds = loss > 0
            return loss, inds
        return loss  # B, N, 3

    def collision_loss_GMoF(self, point_cloud, smpl_output, ret_collision_mask=False, rho=5e-2):
        """
        Args:
            point_cloud (list or torch.tensor): B, N, 3
        """
        sdf_pred = self.query_fast(point_cloud, smpl_output)
        loss = torch.relu(-sdf_pred)  # B,N
        loss = loss ** 2
        loss = torch.div(loss, loss + rho ** 2)
        inds = loss > 0
        loss = loss.sum(-1)/(inds.float().sum(-1)+1e-6)
        if ret_collision_mask is not None:
            return loss, inds
        return loss  # B, N, 3

    @torch.no_grad()
    def color_points(self, pts, smpl_output=None, max_queries=100000):
        part_colors = self.get_part_colors()
        K = self.partitioner.num_parts
        B = pts.shape[0]
        part_colors = part_colors[:K+1]
        part_colors[-1, :] = 0  # set bg to black

        pts_colors = []
        for _v in torch.split(pts, max_queries//B, dim=1):
            part_pred = self.query(_v, smpl_output, ret_intermediate=True)[1]
            label = (part_pred['part_sdf']).argmin(dim=1)  # B,K,V -> B,V
            # label[part_pred['all_out']] = K  # all mlps say outside
            inds = label.reshape(-1).cpu().numpy()
            pts_colors.append(part_colors[inds].reshape((B, -1, 3)))

        pts_colors = np.concatenate(pts_colors, 1)
        return pts_colors

    def query_occupancy(self, points, smpl_output=None, ret_intermediate=False):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        # attach variables
        self._attach_impl_code(smpl_output)
        local_queries, inside_bbox = self.to_local(points, self.impl_code['bone_trans'], self.impl_code['bbox_center'], self.impl_code['bbox_size'])
        occupancy = self._fwd_pass(local_queries, inside_bbox)[0] # B,K
        return occupancy

    def self_collision_loss(self, smpl_output, n_points_uniform=300, at_least_n_samples=2, ret_samples=False):
        b_smpl_output_list = self.batchify_smpl_output(smpl_output)
        self_pen_losses = []
        samples_list = []
        for b_ind in range(len(b_smpl_output_list)):
            loss, _samples, _ = self._self_penetration_loss(b_smpl_output_list[b_ind], n_points_uniform, at_least_n_samples)
            self_pen_losses.append(loss)
            if ret_samples:
                samples_list.append(_samples)
        self_pen_losses = torch.stack(self_pen_losses)
        if ret_samples:
            return self_pen_losses, samples_list
        return self_pen_losses

    def _self_penetration_loss(self, smpl_output, n_points_uniform, at_least_n_samples):
        """
        Args:
            n_points_uniform (int): how many points per collided body part to sample

        Returns:

        """
        self._attach_impl_code(smpl_output)
        bb_min = self.impl_code['bbox_min'].squeeze(0)
        bb_max = self.impl_code['bbox_max'].squeeze(0)
        device = bb_min.device

        with torch.no_grad():
            abs_trans = torch.inverse(self.impl_code['bone_trans']).squeeze(0)
            hcorners, collided_inds = self.partitioner.check_bbox_penetrations(bb_min, bb_max, abs_trans)
            collision_candidates = self.partitioner.sample_collision_candidates(
                bb_min, bb_max, abs_trans, collided_inds, n_points_uniform)

        # remove points that are outside
        if collision_candidates.shape[0] == 0:
            return torch.tensor(0, device=device, dtype=torch.float32), None, None
        collision_candidates = torch.cat((collision_candidates, smpl_output.vertices[0].detach().clone()), dim=0)
        if True:
            local_queries, inside_bbox = self.to_local(collision_candidates.unsqueeze(0), self.impl_code['bone_trans'], self.impl_code['bbox_center'], self.impl_code['bbox_size'])
            per_part_occupancy = self._fwd_pass(local_queries, inside_bbox, only_part_occupancy=True).squeeze(0)  # (K, N)

        with torch.no_grad():
            _po = per_part_occupancy.t() # N, K
            _po_mask = (_po > self.level_set).float()
            disable_mat = (~torch.eye(_po.shape[-1], dtype=torch.bool, device=_po.device))
            disable_mat = (disable_mat & self.partitioner.selfpen_disable_mat).float()
            _conflicting_inds = (_po_mask[:, :, None] @ _po_mask[:, None, :])*disable_mat[None] # N,K,K
            conflicting_inds = _conflicting_inds.reshape(_conflicting_inds.shape[0], -1).sum(-1) >= 2.0
        
        conf_parts = per_part_occupancy.t()[conflicting_inds]  # inside 1, outside 0 [samples, K]
        if len(conf_parts.shape) == 0 or conf_parts.shape[0] == 0:
            return torch.tensor(0, device=device, dtype=torch.float32), None, None

        affected_inds = (conf_parts > self.level_set).sum(0) >= at_least_n_samples
        conf_parts = torch.relu(conf_parts)
        loss = (conf_parts.sum(-1) - self.level_set).sum()
        _samples = collision_candidates[conflicting_inds]
        print(f'\n_samples={_samples.shape[0]} ({collision_candidates.shape[0]})\n', np.array(JOINT_NAMES)[torch.where(self.partitioner.joint_mapper)[0].cpu().detach().numpy()][torch.where(affected_inds)[0].detach().cpu().numpy().tolist()])
        return loss, _samples, affected_inds

@torch.no_grad()
def debug_vis_body_model(parametric_body):
    import pyrender
    def vis_create_pc(pts, color=(0.0, 1.0, 0.0), radius=0.005):
        if torch.is_tensor(pts):
            pts = pts.detach().cpu().numpy()
        pts = pts.reshape((-1, 3))
        tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
        tfs[:, :3, 3] = pts
        sm_in = trimesh.creation.uv_sphere(radius=radius)
        sm_in.visual.vertex_colors = color

        return pyrender.Mesh.from_trimesh(sm_in, poses=tfs)

    partitioner = Partitioner(parametric_body)
 
    template = parametric_body.v_template
    tight_face_tensor = partitioner.tight_face_tensor
    extended_face_tensor = partitioner.extended_face_tensor
    tight_vert_selector = partitioner.tight_vert_selector
    n_parts = tight_face_tensor.shape[0]

    pts = partitioner.sample_mesh_points(template[None], 1000) # B, n_parts, N, 3
    pts = pts.squeeze(0) # n_parts, N, 3
    pts_rad = (template[1:] - template[:-1]).norm(dim=-1).mean().item()*0.1

    for b_part in range(n_parts):
        vertex_colors = np.zeros((template.shape[0], 4), dtype=np.uint8)
        vertex_colors[..., :] = np.array([127,127,127,255], dtype=np.uint8)
        vertex_colors[tight_vert_selector[b_part], :] = np.array([0,0,255,255], dtype=np.uint8)
        print('Part:', b_part, 'Num vertices:', tight_vert_selector[b_part].shape[0])
        template_mesh = trimesh.Trimesh(vertices=template, faces=parametric_body.faces, vertex_colors=vertex_colors)

        parts_pts = pts.clone()
        part_pts = vis_create_pc(parts_pts[b_part], color=(0., 1., 0.), radius=pts_rad)  # green - part points

        scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
        # posed_mesh = get_posed_mesh(posed_vertices, tight_vert_selector, , b_part)
        scene.add(part_pts)
        scene.add(pyrender.Mesh.from_trimesh(template_mesh, smooth=False))
        # for _b_part in range(24):
        #     scene.add(vis_create_pc(posed_samp_cube[0][_b_part], color=(1., 0., 0.), radius=0.0025))  # red - cube
        # scene.add(vis_create_pc(bbox_min_fwd[0][b_part], color=(0., 0., 1.), radius=0.01))
        # scene.add(vis_create_pc(bbox_max_fwd[0][b_part], color=(0., 1., 1.), radius=0.01))
        pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_path', default='/media/STORAGE_4TB/body_models', type=str)
    parser.add_argument('--model_type', default='mano', type=str, choices=['smpl', 'mano', 'smplx'])
    args = parser.parse_args()

    _bm = smplx.create(args.body_path, model_type=args.model_type, batch_size=1)
    debug_vis_body_model(_bm)
