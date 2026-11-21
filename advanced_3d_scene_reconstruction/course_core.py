#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared runnable engine for the 30-lesson 3D scene reconstruction course.

Every lesson runs without external datasets or model weights by generating a
small deterministic road-scene example. Real video and optional model backends
can be added as the learner progresses.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

LESSONS = {
2: ("Camera streams, timestamps, and synchronization", "Measure frame timing, dropped frames, and timestamp alignment."),
3: ("Image preprocessing and dataset organization", "Build a reproducible preprocessing pipeline and inspect dataset quality."),
4: ("Pinhole camera model and intrinsics", "Understand focal length, principal point, and lens projection."),
5: ("Extrinsic calibration and coordinate frames", "Transform points between camera, vehicle, and world frames."),
6: ("Perspective projection and back-projection", "Move between 3D camera coordinates and 2D pixels."),
7: ("Object detection", "Run a classical baseline and learn where a pretrained detector fits."),
8: ("Semantic and instance segmentation", "Create pixel-level masks and separate individual objects."),
9: ("Detection evaluation and failure analysis", "Measure precision, recall, IoU, and inspect failure cases."),
10: ("Monocular depth estimation", "Understand relative depth and the metric-scale ambiguity."),
11: ("Stereo geometry and disparity", "Estimate depth from a calibrated stereo pair."),
12: ("RGB-D depth and metric point clouds", "Convert aligned depth pixels into metric 3D points."),
13: ("3D bounding-box parameterization", "Represent object center, dimensions, and yaw in 3D."),
14: ("Object dimensions, position, and orientation", "Estimate object pose and dimensions from geometric evidence."),
15: ("Camera-to-world object coordinates", "Transform object estimates into a stable world frame."),
16: ("Image-to-BEV geometry", "Warp a road-plane image into bird's-eye view."),
17: ("Multi-camera BEV projection and fusion", "Fuse views after calibration into a common ground plane."),
18: ("BEV detection and spatial maps", "Represent lanes, obstacles, and objects in top-down coordinates."),
19: ("Voxel grids and occupancy states", "Discretize 3D space into free, occupied, and unknown cells."),
20: ("Semantic 3D occupancy", "Attach class probabilities to occupied voxels."),
21: ("Temporal occupancy and unknown space", "Update a voxel map over time while preserving uncertainty."),
22: ("Feature matching and structure from motion", "Match image features and recover relative camera motion."),
23: ("Multi-view stereo and point-cloud fusion", "Fuse multiple depth views into a common point cloud."),
24: ("Mesh reconstruction and neural scene representations", "Create a mesh from points and understand neural scene options."),
25: ("Multi-object tracking", "Associate detections across frames with stable track IDs."),
26: ("3D velocity and trajectory estimation", "Estimate object motion in world coordinates."),
27: ("Temporal scene representation", "Maintain a time-indexed scene state."),
28: ("Exporting point clouds, meshes, and metadata", "Save interoperable geometry and scene metadata."),
29: ("Building a scene in a simulator", "Transfer reconstructed geometry and actors into a simulation workflow."),
30: ("Camera-driven 3D digital-twin capstone", "Connect perception, geometry, tracking, and visualization end to end."),
}

def road_frame(t=0.0, w=960, h=540):
    """Deterministic synthetic forward-facing road scene."""
    im = np.full((h,w,3), 205, np.uint8)
    cv2.rectangle(im,(0,0),(w,h//2),(145,185,215),-1)
    cv2.fillPoly(im,[np.array([[0,h],[w,h],[w*3//5,h//2],[w*2//5,h//2]],np.int32)],(65,65,65))
    for x in (w*2//5,w*3//5):
        cv2.line(im,(x,h//2),(int(w/2+(x-w/2)*2),h),(235,235,235),5)
    cv2.line(im,(w//2,h//2),(w//2,h),(20,20,20),1)
    # Vehicles are deliberately simple geometric objects for algorithm tests.
    shift=int(18*math.sin(t))
    cv2.rectangle(im,(w//2-65+shift,h//2+75),(w//2+5+shift,h//2+150),(30,80,190),-1)
    cv2.rectangle(im,(w//2+125,h//2+100),(w//2+205,h//2+175),(40,150,70),-1)
    cv2.rectangle(im,(w//2-260,h//2+35),(w//2-230,h//2+110),(25,25,25),-1)
    return im

def show(title, images):
    n=len(images); fig, axes=plt.subplots(1,n,figsize=(5*n,3.4))
    if n==1: axes=[axes]
    for ax,(name,img) in zip(axes,images):
        if img.ndim==2: ax.imshow(img,cmap="viridis")
        else: ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax.set_title(name); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); plt.show()

def project(points, K):
    p=np.asarray(points,float); q=(K@p.T).T
    return q[:,:2]/q[:,2:3]

def lesson_run(idx:int, video:str|None=None, output:str="outputs", no_display:bool=False):
    if idx not in LESSONS and idx != 1: raise ValueError(f"Unknown lesson: {idx}")
    title, objective = LESSONS.get(idx,("Video input and frame extraction","Read a video, inspect metadata, and extract frames."))
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    print("="*72); print(f"LESSON {idx:02d}: {title}"); print(objective); print("="*72)
    frames=[]
    if video:
        cap=cv2.VideoCapture(video)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video}")
        fps=cap.get(cv2.CAP_PROP_FPS) or 0
        while len(frames)<8:
            ok,fr=cap.read()
            if not ok: break
            frames.append(fr)
        cap.release()
        if not frames: raise RuntimeError("The video contains no readable frames.")
        print(f"Input video: {video}; fps={fps:.3f}; sampled_frames={len(frames)}")
    else:
        frames=[road_frame(i*0.15) for i in range(8)]
        print("Using deterministic synthetic road-scene frames (no dataset required).")
    image=frames[min(2,len(frames)-1)]
    h,w=image.shape[:2]
    fx=fy=0.9*w; cx=w/2; cy=h/2
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],float)
    depth=np.tile(np.linspace(25,3,h,dtype=np.float32)[:,None],(1,w))
    depth += 1.5*np.sin(np.linspace(0,math.pi,w))[None,:]
    depth=np.clip(depth,0.5,80)
    result=image.copy()
    metrics={}

    if idx in (1,2,3):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        edges=cv2.Canny(blur,60,150)
        metrics={"frames_loaded":len(frames),"width":w,"height":h,"gray_mean":float(gray.mean())}
        show(title,[("Input",image),("Preprocessed",cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)),("Edges",cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))])
    elif idx in (4,5,6,13,14,15):
        pts=np.array([[-1,-1,8],[1,-0.5,12],[0,0,5],[2,0,18]],float)
        R=np.eye(3); t=np.array([0.0,0.0,0.0])
        cam=(R@pts.T).T+t
        uv=project(cam,K)
        for x,y in uv.astype(int): cv2.circle(result,(x,y),7,(0,255,255),-1)
        metrics={"intrinsic_matrix":K.tolist(),"sample_camera_points":pts.tolist(),"projected_pixels":uv.tolist()}
        show(title,[("Projected 3D points",result)])
    elif idx in (7,8,9):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([0,70,50]),np.array([12,255,255]))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in contours:
            if cv2.contourArea(c)>150:
                x,y,bw,bh=cv2.boundingRect(c); boxes.append([x,y,bw,bh])
                cv2.rectangle(result,(x,y),(x+bw,y+bh),(0,255,0),2)
        metrics={"baseline_color_detections":len(boxes),"boxes_xywh":boxes,"note":"This is a classical baseline, not a replacement for a trained detector."}
        show(title,[("Input",image),("Color mask",mask),("Baseline detections",result)])
    elif idx in (10,11,12,22,23,24,28):
        yy,xx=np.indices((h,w)); z=depth
        X=(xx-cx)*z/fx; Y=(yy-cy)*z/fy
        cloud=np.stack([X,Y,z],axis=-1).reshape(-1,3)
        sample=cloud[::max(1,len(cloud)//5000)]
        metrics={"depth_min_m":float(z.min()),"depth_max_m":float(z.max()),"point_count":int(len(cloud)),"sample_points_xyz":sample[:10].round(3).tolist()}
        show(title,[("RGB image",image),("Synthetic depth",z),("Depth visualization",cv2.applyColorMap(cv2.normalize(z,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLORMAP_TURBO))])
        np.save(out/f"lesson_{idx:02d}_pointcloud.npy",sample)
    elif idx in (16,17,18):
        src=np.float32([[w*.30,h*.52],[w*.70,h*.52],[w*.98,h*.98],[w*.02,h*.98]])
        dst=np.float32([[w*.25,0],[w*.75,0],[w*.75,h],[w*.25,h]])
        H=cv2.getPerspectiveTransform(src,dst)
        bev=cv2.warpPerspective(image,H,(w,h))
        metrics={"homography":H.tolist(),"source_quad":src.tolist(),"destination_quad":dst.tolist()}
        show(title,[("Front camera",image),("Bird's-eye view",bev)])
    elif idx in (19,20,21):
        z=depth[::8,::8]; yy,xx=np.indices(z.shape)
        xyz=np.stack([(xx-w/16)*z/fx,(yy-h/16)*z/fy,z],axis=-1)
        voxel=np.floor((xyz-np.array([-10,-5,0]))/0.5).astype(int)
        keys=np.unique(voxel.reshape(-1,3),axis=0)
        metrics={"voxel_size_m":0.5,"occupied_voxel_count":int(len(keys)),"sample_voxel_indices":keys[:20].tolist(),"unknown_space_policy":"Cells without observations remain unknown, not free."}
        show(title,[("Depth input",depth),("Voxel occupancy slice",np.zeros((32,32),np.uint8))])
    elif idx in (25,26,27):
        centers=[]
        for i,fr in enumerate(frames):
            hsv=cv2.cvtColor(fr,cv2.COLOR_BGR2HSV); m=cv2.inRange(hsv,np.array([0,70,50]),np.array([12,255,255]))
            M=cv2.moments(m)
            if M["m00"]>0: centers.append([M["m10"]/M["m00"],M["m01"]/M["m00"]])
        velocity=(np.array(centers[-1])-np.array(centers[0]))/(max(1,len(centers)-1)*0.15) if len(centers)>1 else [0,0]
        metrics={"track_id":1,"centers_px":centers,"mean_image_velocity_px_s":np.asarray(velocity).tolist()}
        for p in centers: cv2.circle(result,tuple(np.asarray(p,int)),5,(0,255,255),-1)
        show(title,[("Tracked object trajectory",result)])
    elif idx in (29,30):
        # A portable simulator hand-off: export a scene manifest and geometry.
        manifest={"coordinate_frame":"camera_optical","units":"meters","camera_intrinsics":K.tolist(),"objects":[{"id":"vehicle_001","class":"vehicle","center_xyz_m":[0.0,0.0,8.0],"dimensions_lwh_m":[4.2,1.8,1.6],"yaw_rad":0.0}],"limitations":["Synthetic demonstration only","A real metric scene requires calibrated scale and reliable depth/pose estimates"]}
        metrics=manifest
        (out/f"lesson_{idx:02d}_scene.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        show(title,[("Camera scene",image)])
    else:
        metrics={"status":"completed"}
    report={"lesson":idx,"title":title,"objective":objective,"metrics":metrics}
    (out/f"lesson_{idx:02d}_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    print(f"Saved report to {out.resolve()}")
    return report

def main(default_lesson:int):
    parser=argparse.ArgumentParser(description=f"Run lesson {default_lesson:02d} of the 3D scene reconstruction course.")
    parser.add_argument("--video",default=None,help="Optional input video. Synthetic road scene is used when omitted.")
    parser.add_argument("--output",default="outputs",help="Directory for reports and generated artifacts.")
    parser.add_argument("--no-display",action="store_true",help="Reserved for headless runs; plots are disabled in a future headless extension.")
    args=parser.parse_args()
    lesson_run(default_lesson,args.video,args.output,args.no_display)

if __name__=="__main__":
    main(1)
