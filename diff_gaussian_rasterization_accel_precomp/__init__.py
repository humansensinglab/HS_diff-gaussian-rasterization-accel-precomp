#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import nvtx
import numpy as np
import sys
def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def validate_and_print_arguments(*args):
    for i, arg in enumerate(args):
        if isinstance(arg, np.ndarray):
            print(f"Argument {i}: numpy array, shape: {arg.shape}, dtype: {arg.dtype}")
        elif isinstance(arg, torch.Tensor):
            print(f"Argument {i}: torch tensor, shape: {arg}, dtype: {arg.dtype}")
        else:
            print(f"Argument {i}: {arg}")

def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings
    ):

        
        args = (
                raster_settings.bg, 
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                dc,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                raster_settings.debug,
                raster_settings.store_ordering,
                raster_settings.using_precomp,
                raster_settings.gaussian_list,
                raster_settings.ranges,
                raster_settings.num_buck,
                raster_settings.num_rend,
                raster_settings.graphable,
                raster_settings.img_buffer,
                raster_settings.geom_buffer,
                raster_settings.sample_buffer)
        

        if raster_settings.using_precomp:
               
               #validate_and_print_arguments(*args)
               #print("i")
               #r = nvtx.start_range("precomp")
               
            

            num_rendered, num_buckets,  ranges, gaussian_list, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)   
               
            
               


            #nvtx.end_range(r)
            ctx.raster_settings = raster_settings
            ctx.num_rendered = num_rendered
            ctx.num_buckets = num_buckets
            ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, raster_settings.geom_buffer, binningBuffer, raster_settings.img_buffer, raster_settings.sample_buffer)
            return color, radii, None, None,None, num_rendered,0,0,0,0
        else:
            # Invoke C++/CUDA rasterizer
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    num_rendered, num_buckets, ranges, gaussian_list ,color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                #validate_and_print_arguments(*args)
                r = nvtx.start_range("Non precomp")
                num_rendered, num_buckets,  ranges, gaussian_list, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)
                nvtx.end_range(r)
                
           
            # Keep relevant tensors for backward
            ctx.raster_settings = raster_settings
            ctx.num_rendered = num_rendered
            ctx.num_buckets = num_buckets
            ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)
            if raster_settings.store_ordering:
                return color, radii, gaussian_list, ranges, None,num_rendered, num_buckets, imgBuffer.size(), geomBuffer.size(), sampleBuffer.size()
            return color, radii,  None, None, None,num_rendered, 0,0,0,0

    @staticmethod
    def backward(ctx, grad_out_color, *_):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                dc,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                raster_settings.debug,
                raster_settings.using_precomp,
                raster_settings.graphable,
                raster_settings.gaussian_list,
                raster_settings.ranges,
                raster_settings.dL_dmeans3D,
                raster_settings.dL_dmeans2D,
                raster_settings.dL_dcolors,
                raster_settings.dL_dconic,
                raster_settings.dL_dopacity,
                raster_settings.dL_dcov3D,
                raster_settings.dL_ddc,
                raster_settings.dL_dsh,
                raster_settings.dL_dscales,
                raster_settings.dL_drotations)
        
        #validate_and_print_arguments(*args)
        
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             r = nvtx.start_range("backward")
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
             nvtx.end_range(r)
                
                   
        if(not raster_settings.graphable):
        # if(1==1):
            grads = (
                grad_means3D,
                grad_means2D,
                grad_dc,
                grad_sh,
                grad_colors_precomp,
                grad_opacities,
                grad_scales,
                grad_rotations,
                grad_cov3Ds_precomp,
                None,
            )
        else:  
           
            grads= (
                raster_settings.dL_dmeans3D,
                raster_settings.dL_dmeans2D,
                raster_settings.dL_ddc,
                raster_settings.dL_dsh,
                raster_settings.dL_dcolors,
                raster_settings.dL_dopacity,
                raster_settings.dL_dscales,
                raster_settings.dL_drotations,
                raster_settings.dL_dcov3D,
                None
         ) 
        


        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    store_ordering : bool
    using_precomp : bool
    gaussian_list : torch.Tensor
    ranges : torch.Tensor
    num_buck : int
    num_rend : int
    graphable : bool
    img_buffer : torch.Tensor
    geom_buffer : torch.Tensor
    sample_buffer : torch.Tensor
    dL_dmeans3D : torch.Tensor
    dL_dmeans2D : torch.Tensor
    dL_dcolors : torch.Tensor 
    dL_dconic : torch.Tensor
    dL_dopacity : torch.Tensor 
    dL_dcov3D : torch.Tensor
    dL_ddc : torch.Tensor
    dL_dsh : torch.Tensor
    dL_dscales : torch.Tensor
    dL_drotations : torch.Tensor
         

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings
       
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings
        )

