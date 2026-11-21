#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED RESEARCH TRACK — executable mathematical laboratories.

IMPORTANT:
- These are transparent reference implementations for learning geometry.
- They are NOT reimplementations of VGGT, SAM 3.1, D²USt3R, 3DGS, or a
  production autonomous-driving occupancy network.
- Real research models are integrated through adapters in later experiments.

Core conventions:
- Camera coordinates: +X right, +Y down, +Z forward (OpenCV convention).
- Distances: meters whenever the input depth is metric.
- Transform notation: T_A_B maps coordinates expressed in frame B into frame A.
- A point cloud without a declared frame, unit, timestamp, and confidence is
  incomplete engineering data.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

W, H = 640, 360
K = np.array([[520.0, 0.0, W/2], [0.0, 520.0, H/2], [0.0, 0.0, 1.0]], dtype=np.float64)

def make_scene(t: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic RGB, metric depth (m), and a moving-object mask.

    The synthetic scene is deliberately simple so geometry tests have known
    ground truth. It is a teaching fixture, not a photorealistic dataset.
    """
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (175, 190, 205)
    cv2.rectangle(img, (0, H//2), (W-1, H-1), (65, 65, 65), -1)
    cv2.line(img, (W//2, H//2), (W//2, H-1), (230,230,230), 3)
    depth = np.full((H, W), 18.0, np.float32)
    depth[H//2:, :] = np.linspace(15.0, 4.0, H-H//2, dtype=np.float32)[:,None]
    # A foreground car moves laterally. Its depth is known by construction.
    x = int(W*0.47 + 45*np.sin(t))
    box = (x-32, 205, x+32, 280)
    cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (25,55,210), -1)
    mask = np.zeros((H,W), np.uint8)
    cv2.rectangle(mask, (box[0],box[1]), (box[2],box[3]), 255, -1)
    depth[mask>0] = 6.0
    # Add lane markings and texture to support feature/flow demonstrations.
    for yy in range(190, H, 28):
        cv2.circle(img, (W//2, yy), 2, (255,255,255), -1)
    rng = np.random.default_rng(7)
    for _ in range(120):
        px, py = int(rng.integers(5,W-5)), int(rng.integers(5,H//2))
        cv2.circle(img,(px,py),1,(130,140,150),-1)
    return img, depth, mask

def backproject(depth: np.ndarray, Kmat: np.ndarray = K) -> np.ndarray:
    """Lift each pixel into camera-frame XYZ using Z-depth.

    x=(u-cx)Z/fx and y=(v-cy)Z/fy. This assumes depth stores optical-axis
    Z, not Euclidean range along the viewing ray.
    """
    h,w=depth.shape
    v,u=np.indices((h,w),dtype=np.float64)
    z=depth.astype(np.float64)
    x=(u-Kmat[0,2])*z/Kmat[0,0]
    y=(v-Kmat[1,2])*z/Kmat[1,1]
    return np.stack((x,y,z),axis=-1)

def transform_points(points: np.ndarray, T_A_B: np.ndarray) -> np.ndarray:
    """Apply a homogeneous rigid transform; validates SE(3)-shaped input."""
    if T_A_B.shape != (4,4): raise ValueError("Transform must be 4x4.")
    p=np.asarray(points).reshape(-1,3)
    ph=np.c_[p,np.ones(len(p))]
    return (T_A_B@ph.T).T[:,:3]

def fit_depth_scale_shift(pred: np.ndarray, target: np.ndarray, valid: np.ndarray|None=None):
    """Least-squares fit target ~= scale * pred + shift.

    This aligns a relative-depth prediction to metric anchors. It does not
    make the monocular prediction metric without trustworthy anchors.
    """
    p=np.asarray(pred,dtype=np.float64).ravel()
    y=np.asarray(target,dtype=np.float64).ravel()
    m=np.isfinite(p)&np.isfinite(y)
    if valid is not None: m &= np.asarray(valid).ravel().astype(bool)
    if m.sum()<2: raise ValueError("At least two valid depth anchors are required.")
    A=np.c_[p[m],np.ones(m.sum())]
    scale,shift=np.linalg.lstsq(A,y[m],rcond=None)[0]
    residual=y[m]-(scale*p[m]+shift)
    return float(scale),float(shift),float(np.sqrt(np.mean(residual**2)))

def gaussian_splat_2d(canvas: np.ndarray, means: np.ndarray, covariances: np.ndarray,
                      colors: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """Tiny 2D Gaussian compositor for understanding projected 3DGS primitives.

    A 3DGS renderer projects a 3D covariance into screen space, evaluates a
    2D Gaussian footprint, sorts primitives approximately front-to-back, and
    alpha-composites colors. This pedagogical renderer omits SH appearance,
    tile-based rasterization, gradients, and visibility optimizations.
    """
    out=canvas.astype(np.float32).copy()
    h,w=out.shape[:2]
    order=np.argsort(means[:,2])[::-1]  # far-to-near for simple over compositing
    yy,xx=np.mgrid[0:h,0:w]
    for i in order:
        mu=means[i,:2]; cov=covariances[i]
        inv=np.linalg.inv(cov+np.eye(2)*1e-5)
        dx=xx-mu[0]; dy=yy-mu[1]
        exponent=inv[0,0]*dx*dx+2*inv[0,1]*dx*dy+inv[1,1]*dy*dy
        a=np.clip(opacity[i]*np.exp(-0.5*exponent),0,1)[...,None]
        out=out*(1-a)+colors[i][None,None,:]*a
    return np.clip(out,0,255).astype(np.uint8)

def build_occupancy(points: np.ndarray, voxel: float=0.5, bounds=(-12,-8,0,12,8,25)):
    """Quantize XYZ points into integer voxel coordinates and unique cells.

    Occupied cells are supported by returns. Empty cells are NOT automatically
    free: a ray traversal/sensor model is required to label observed free space.
    """
    xmin,ymin,zmin,xmax,ymax,zmax=bounds
    p=np.asarray(points).reshape(-1,3)
    inside=(p[:,0]>=xmin)&(p[:,0]<xmax)&(p[:,1]>=ymin)&(p[:,1]<ymax)&(p[:,2]>=zmin)&(p[:,2]<zmax)
    idx=np.floor((p[inside]-[xmin,ymin,zmin])/voxel).astype(np.int32)
    return np.unique(idx,axis=0)

def show(title, panels):
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),3.5))
    if len(panels)==1: axes=[axes]
    for ax,(name,img) in zip(axes,panels):
        if img.ndim==2: ax.imshow(img,cmap="viridis")
        else: ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax.set_title(name); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); plt.show()

def run_lesson(n:int, output="outputs", no_display=False):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    img,depth,dynamic=make_scene(0.6)
    xyz=backproject(depth)
    report={"lesson":n,"coordinate_frame":"camera_optical","units":"meters","note":"Synthetic mathematical lab; not a pretrained foundation-model inference."}
    panels=[]
    if n==31:
        # VGGT-like output contract: per-frame camera, depth, point map, tracks.
        # Here we synthesize the point map from known depth to test conventions.
        uv=np.array([[W/2,H/2],[W/2+50,H/2+20]],float)
        z=np.array([5.,10.])
        pts=np.c_[(uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z]
        report.update({"point_map_shape":list(xyz.shape),"sample_xyz_m":pts.tolist(),"camera_intrinsics":K.tolist(),"model_adapter":"Connect official VGGT inference separately."})
        panels=[("RGB",img),("Metric depth (synthetic)",depth)]
    elif n==32:
        # Dynamic points violate the static-world assumption used by ordinary
        # multi-view alignment. Keep dynamic masks separate during alignment.
        report.update({"static_pixel_count":int((dynamic==0).sum()),"dynamic_pixel_count":int((dynamic>0).sum()),"recommendation":"Estimate ego-motion on static/inlier regions; model moving objects independently."})
        panels=[("RGB",img),("Dynamic-region mask",dynamic)]
    elif n==33:
        pred=1.0/(depth+1e-3)  # deliberately relative-like monotonic prediction
        anchors=np.zeros_like(depth,dtype=bool); anchors[::45,::45]=True
        s,b,rmse=fit_depth_scale_shift(pred,depth,anchors)
        report.update({"scale":s,"shift":b,"anchor_rmse_m":rmse,"anchor_count":int(anchors.sum()),"warning":"Scale/shift calibration is only as good as the metric anchors."})
        panels=[("Relative-like prediction",pred),("Metric anchor depth",depth)]
    elif n==34:
        # Color threshold is only a placeholder segmentation baseline.
        # Replace this block with SAM 3.1 promptable concept segmentation.
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([0,80,50]),np.array([12,255,255]))
        report.update({"baseline_mask_pixels":int((mask>0).sum()),"foundation_model":"SAM 3 / SAM 3.1 adapter not loaded","prompt_example":"red vehicle"})
        panels=[("RGB",img),("Classical baseline mask",mask)]
    elif n==35:
        prev=cv2.cvtColor(make_scene(0.2)[0],cv2.COLOR_BGR2GRAY)
        curr=cv2.cvtColor(make_scene(0.6)[0],cv2.COLOR_BGR2GRAY)
        p0=cv2.goodFeaturesToTrack(prev,maxCorners=250,qualityLevel=.01,minDistance=7)
        p1,st,err=cv2.calcOpticalFlowPyrLK(prev,curr,p0,None)
        good=(st.ravel()==1)
        vis=img.copy()
        for a,b in zip(p0[good],p1[good]):
            cv2.line(vis,tuple(a.ravel().astype(int)),tuple(b.ravel().astype(int)),(0,255,0),1)
            cv2.circle(vis,tuple(b.ravel().astype(int)),2,(0,0,255),-1)
        report.update({"tracked_points":int(good.sum()),"method":"Pyramidal Lucas-Kanade baseline","warning":"2D tracks require depth and camera poses to become 3D tracks."})
        panels=[("Tracked points",vis)]
    elif n==36:
        # Motion segmentation is not identical to raw optical flow:
        # compensate camera/ego motion before interpreting residual flow.
        report.update({"dynamic_mask_area_px":int((dynamic>0).sum()),"static_dynamic_separation":"Synthetic ground-truth mask","real_pipeline":["estimate ego pose","warp previous frame","compute residual motion","fuse semantic and geometric cues"]})
        panels=[("RGB",img),("Dynamic mask",dynamic)]
    elif n in (37,38):
        # Projected Gaussian means and covariances are a compact introduction
        # to 3DGS. Full 3D covariance projection uses Jacobian J:
        # Sigma_2D = J W Sigma_3D W^T J^T.
        means=np.array([[180,190,10],[320,230,6],[460,190,14]],float)
        covs=np.array([[[260,0],[0,90]],[[420,40],[40,150]],[[180,0],[0,70]]],float)
        colors=np.array([[30,100,230],[30,220,70],[230,160,20]],float)
        opacity=np.array([.8,.75,.7])
        if n==38:
            means[:,0]+=np.array([-20,25,45]) # time-conditioned motion demo
        rendered=gaussian_splat_2d(np.zeros_like(img),means,covs,colors,opacity)
        report.update({"gaussian_count":3,"means_xy_depth":means.tolist(),"covariance_2d":covs.tolist(),"lesson_scope":"2D projected Gaussian teaching renderer, not full 3DGS/4DGS."})
        panels=[("Gaussian primitives",rendered)]
    elif n==39:
        # Online mapping systems must couple tracking and map update. This lab
        # exports a single keyframe cloud; loop closure is a separate module.
        cloud=xyz[::8,::8].reshape(-1,3)
        report.update({"keyframe_point_count":int(len(cloud)),"map_frame":"camera_optical","slam_components":["visual odometry","keyframe selection","local mapping","loop closure","relocalization"]})
        np.save(out/"lesson39_keyframe_cloud_xyz_m.npy",cloud.astype(np.float32))
        panels=[("Keyframe RGB",img),("Depth",depth)]
    elif n in (40,41,42):
        cloud=xyz[::4,::4].reshape(-1,3)
        vox=build_occupancy(cloud,voxel=.5)
        report.update({"occupied_voxels":int(len(vox)),"voxel_size_m":.5,"occupancy_semantics":"Observed occupied only; unknown and free are not conflated."})
        if n==41:
            report["forecast"]="Constant-velocity baseline placeholder; learned action-conditioned world model is a separate adapter."
            report["forecast_horizon_s"]=[.5,1.0,2.0]
        if n==42:
            manifest={"timestamp_synthetic":0.6,"coordinate_frame":"camera_optical","units":"meters","camera_intrinsics":K.tolist(),"occupied_voxel_count":int(len(vox)),"dynamic_object_pixels":int((dynamic>0).sum())}
            (out/"lesson42_world_model_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            np.save(out/"lesson42_occupied_voxels.npy",vox)
        panels=[("RGB",img),("Depth",depth)]
    else:
        raise ValueError(f"Unsupported advanced lesson {n}")
    (out/f"lesson{n:02d}_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not no_display: show(f"Advanced Lesson {n}",panels)
    print(f"Artifacts saved under: {out.resolve()}")
    return report

def main(default_lesson):
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="outputs")
    p.add_argument("--no-display",action="store_true")
    a=p.parse_args()
    run_lesson(default_lesson,a.output,a.no_display)

if __name__=="__main__":
    main(31)
