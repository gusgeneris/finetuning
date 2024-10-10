import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


from pathlib import Path
import cv2
import trimesh
import nvdiffrast.torch as dr

from model.archs.decoders.shape_texture_net import TetTexNet
from model.archs.unet import UNetPP
from util.renderer import Renderer
from model.archs.mlp_head import SdfMlp, RgbMlp
import xatlas
import torch.nn.functional as F


class Dummy:
    pass

class CRM(nn.Module):
    def __init__(self, specs):

        super(CRM, self).__init__()

        self.specs = specs

        # configs
        input_specs = specs["Input"]
        self.input = Dummy()
        self.input.scale = input_specs['scale']
        self.input.resolution = input_specs['resolution']
        self.tet_grid_size = input_specs['tet_grid_size']
        self.camera_angle_num = input_specs['camera_angle_num']

        self.arch = Dummy()
        self.arch.fea_concat = specs["ArchSpecs"]["fea_concat"]
        self.arch.mlp_bias = specs["ArchSpecs"]["mlp_bias"]

        self.dec = Dummy()
        self.dec.c_dim = specs["DecoderSpecs"]["c_dim"]
        self.dec.plane_resolution = specs["DecoderSpecs"]["plane_resolution"]

        self.geo_type = specs["Train"].get("geo_type", "flex") # "dmtet" or "flex"

        self.unet2 = UNetPP(in_channels=self.dec.c_dim)


        mlp_chnl_s = 3 if self.arch.fea_concat else 1  # 3 for queried triplane feature concatenation
        self.decoder = TetTexNet(plane_reso=self.dec.plane_resolution, fea_concat=self.arch.fea_concat)

        if self.geo_type == "flex":
            self.weightMlp = nn.Sequential(
                            nn.Linear(mlp_chnl_s * 32 * 8, 512),
                            nn.SiLU(),
                            nn.Linear(512, 21))         
            
        self.sdfMlp = SdfMlp(mlp_chnl_s * 32, 512, bias=self.arch.mlp_bias) 
        self.rgbMlp = RgbMlp(mlp_chnl_s * 32, 512, bias=self.arch.mlp_bias)
        self.renderer = Renderer(tet_grid_size=self.tet_grid_size, camera_angle_num=self.camera_angle_num,
                                 scale=self.input.scale, geo_type = self.geo_type)


        self.spob = True if specs['Pretrain']['mode'] is None else False  # whether to add sphere
        self.radius = specs['Pretrain']['radius']  # used when spob

        self.denoising = True
        from diffusers import DDIMScheduler
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")


            # Configuración de UNet++
        self.unet2 = UNetPP(in_channels=self.dec.c_dim)  # Configurar UNet++


        device = torch.device("cuda")

            # Mueve tu modelo al dispositivo
        self.unet2 = self.unet2.to(device)
        print(f"UNet++ configurado con in_channels: {self.dec.c_dim}")

        # Decodificador
        self.decoder = TetTexNet(plane_reso=self.dec.plane_resolution, fea_concat=self.arch.fea_concat)
        print(f"Decodificador configurado con plane_resolution: {self.dec.plane_resolution}")

        # MLP para SDF
        self.sdfMlp = SdfMlp(mlp_chnl_s * 32, 512, bias=self.arch.mlp_bias)
        print(f"MLP para SDF configurado con canales: {mlp_chnl_s * 32}")

        # MLP para colores RGB
        self.rgbMlp = RgbMlp(mlp_chnl_s * 32, 512, bias=self.arch.mlp_bias)
        print(f"MLP para RGB configurado con canales: {mlp_chnl_s * 32}")

        if self.geo_type == "flex":
            self.weightMlp = nn.Sequential(
                nn.Linear(mlp_chnl_s * 32 * 8, 512),
                nn.SiLU(),
                nn.Linear(512, 21)
            )  # MLP adicional para el cálculo de pesos si el tipo de geometría es flexible
            print(f"MLP de pesos configurado con entrada de tamaño: {mlp_chnl_s * 32 * 8}")

        # print(f"Dimensiones de inputs: {inputs.size()}")
#             import torch
# import torch.nn.functional as F

    def load_weights(self, model_path):
        device = torch.device("cuda" )
        # Cargar pesos y mover a GPU
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)  # Mover el modelo al dispositivo

    
    def forward(self, inputs):
        print('zasz')
        print(f"Input shape: {inputs.shape}")

        # Redimensionar inputs a 256x256 si es necesario
        if inputs.size(2) != 256 or inputs.size(3) != 256:
            print(f"Redimensionando inputs de {inputs.size(2)}x{inputs.size(3)} a 256x256")
            inputs = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)

        try:
            features = self.unet2(inputs)
            print(f"Features shape after unet2: {features.shape}")

        except Exception as e:
            print(f"Error in self.unet2: {e}")
            return None

        x = features

        # Inicializa learned_plane con las dimensiones correctas
        learned_plane = torch.randn(x.size(0), 32, x.size(2), x.size(3))  # Asegúrate de que learned_plane tenga la misma altura y anchura que x


        device = torch.device("cuda")
        learned_plane = learned_plane.to(device)

        # # Mover learned_plane al mismo dispositivo que el modelo
        # learned_plane = learned_plane.to(x.device)


        print(f"Dispositivo de learned_plane: {learned_plane.device}")

        print(f"x size: {x.size()}, learned_plane size: {learned_plane.size()}")

        # Asegúrate de que learned_plane tenga las dimensiones correctas
        if x.size(2) != learned_plane.size(2) or x.size(3) != learned_plane.size(3):
            learned_plane = F.interpolate(learned_plane, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Revisa las dimensiones después de la interpolación
        print(f"learned_plane size after interpolation: {learned_plane.size()}")

        # Concatenar
        try:
            x = torch.cat([x, learned_plane], dim=1)
            print(f"Concatenated x size: {x.size()}")
        except Exception as e:
            print(f"Error in concatenation: {e}")
            return None

        # Resto del procesamiento...
        verts = self.decoder(features)
        print(f"verts size: {verts.size()}")  # Verifica las dimensiones de verts
        sdf_outputs = self.sdfMlp(verts)
        print(f"sdf_outputs size: {sdf_outputs.size()}")  # Verifica las dimensiones de sdf_outputs
        pred_sdf, deformation = sdf_outputs[..., 0], sdf_outputs[..., 1:]
        rendered_output = self.renderer(inputs, pred_sdf, deformation, verts)

        return rendered_output





    def decode(self, data, triplane_feature2):
        if self.geo_type == "flex":
            tet_verts = self.renderer.flexicubes.verts.unsqueeze(0)
            tet_indices = self.renderer.flexicubes.indices

        dec_verts = self.decoder(triplane_feature2, tet_verts)
        out = self.sdfMlp(dec_verts)

        weight = None
        if self.geo_type == "flex":
            grid_feat = torch.index_select(input=dec_verts, index=self.renderer.flexicubes.indices.reshape(-1),dim=1)
            grid_feat = grid_feat.reshape(dec_verts.shape[0], self.renderer.flexicubes.indices.shape[0], self.renderer.flexicubes.indices.shape[1] * dec_verts.shape[-1])
            weight = self.weightMlp(grid_feat)
            weight = weight * 0.1

        pred_sdf, deformation = out[..., 0], out[..., 1:]
        if self.spob:
            pred_sdf = pred_sdf + self.radius - torch.sqrt((tet_verts**2).sum(-1))

        _, verts, faces = self.renderer(data, pred_sdf, deformation, tet_verts, tet_indices, weight= weight)
        return verts[0].unsqueeze(0), faces[0].int()

    def export_mesh(self, data, out_dir, ind, device=None, tri_fea_2 = None):
        verts = data['verts']
        faces = data['faces']

        dec_verts = self.decoder(tri_fea_2, verts.unsqueeze(0))
        colors = self.rgbMlp(dec_verts).squeeze().detach().cpu().numpy()
        # Expect predicted colors value range from [-1, 1]
        colors = (colors * 0.5 + 0.5).clip(0, 1)

        verts = verts.squeeze().cpu().numpy()
        faces = faces[..., [2, 1, 0]].squeeze().cpu().numpy()

        # export the final mesh
        with torch.no_grad():
            mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors, process=False) # important, process=True leads to seg fault...
            mesh.export(out_dir / f'{ind}.obj')

    def export_mesh_wt_uv(self, ctx, data, out_dir, ind, device, res, tri_fea_2=None):

        mesh_v = data['verts'].squeeze().cpu().numpy()
        mesh_pos_idx = data['faces'].squeeze().cpu().numpy()

        def interpolate(attr, rast, attr_idx, rast_db=None):
            return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                                  diff_attrs=None if rast_db is None else 'all')

        vmapping, indices, uvs = xatlas.parametrize(mesh_v, mesh_pos_idx)

        mesh_v = torch.tensor(mesh_v, dtype=torch.float32, device=device)
        mesh_pos_idx = torch.tensor(mesh_pos_idx, dtype=torch.int64, device=device)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
        mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
        # mesh_v_tex. ture
        uv_clip = uvs[None, ...] * 2.0 - 1.0

        # pad to four component coordinate
        uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

        # rasterize
        rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), res)

        # Interpolate world space position
        gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
        mask = rast[..., 3:4] > 0

        # return uvs, mesh_tex_idx, gb_pos, mask
        gb_pos_unsqz = gb_pos.view(-1, 3)
        mask_unsqz = mask.view(-1)
        tex_unsqz = torch.zeros_like(gb_pos_unsqz) + 1

        gb_mask_pos = gb_pos_unsqz[mask_unsqz]

        gb_mask_pos = gb_mask_pos[None, ]

        with torch.no_grad():

            dec_verts = self.decoder(tri_fea_2, gb_mask_pos)
            colors = self.rgbMlp(dec_verts).squeeze()

        # Expect predicted colors value range from [-1, 1]
        lo, hi = (-1, 1)
        colors = (colors - lo) * (255 / (hi - lo))
        colors = colors.clip(0, 255)

        tex_unsqz[mask_unsqz] = colors

        tex = tex_unsqz.view(res + (3,))

        verts = mesh_v.squeeze().cpu().numpy()
        faces = mesh_pos_idx[..., [2, 1, 0]].squeeze().cpu().numpy()
        # faces = mesh_pos_idx
        # faces = faces.detach().cpu().numpy()
        # faces = faces[..., [2, 1, 0]]
        indices = indices[..., [2, 1, 0]]

        # xatlas.export(f"{out_dir}/{ind}.obj", verts[vmapping], indices, uvs)
        matname = f'{out_dir}.mtl'
        # matname = f'{out_dir}/{ind}.mtl'
        fid = open(matname, 'w')
        fid.write('newmtl material_0\n')
        fid.write('Kd 1 1 1\n')
        fid.write('Ka 1 1 1\n')
        # fid.write('Ks 0 0 0\n')
        fid.write('Ks 0.4 0.4 0.4\n')
        fid.write('Ns 10\n')
        fid.write('illum 2\n')
        fid.write(f'map_Kd {out_dir.split("/")[-1]}.png\n')
        fid.close()

        fid = open(f'{out_dir}.obj', 'w')
        # fid = open(f'{out_dir}/{ind}.obj', 'w')
        fid.write('mtllib %s.mtl\n' % out_dir.split("/")[-1])

        for pidx, p in enumerate(verts):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[2], - pp[1]))

        for pidx, p in enumerate(uvs):
            pp = p
            fid.write('vt %f %f\n' % (pp[0], 1 - pp[1]))

        fid.write('usemtl material_0\n')
        for i, f in enumerate(faces):
            f1 = f + 1
            f2 = indices[i] + 1
            fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
        fid.close()

        img = np.asarray(tex.data.cpu().numpy(), dtype=np.float32)
        mask = np.sum(img.astype(float), axis=-1, keepdims=True)
        mask = (mask <= 3.0).astype(float)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        img = img * (1 - mask) + dilate_img * mask
        img = img.clip(0, 255).astype(np.uint8)

        cv2.imwrite(f'{out_dir}.png', img[..., [2, 1, 0]])
        # cv2.imwrite(f'{out_dir}/{ind}.png', img[..., [2, 1, 0]])
