import math
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import  render_sphere
import torchvision
from scene.cameras import  SphereCam
from scene.colmap_loader import qvec2rotmat
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def angle_axis_to_quaternion(angle_axis: np.ndarray):
    angle = np.linalg.norm(angle_axis)

    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle

    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)

    return np.array([qw, qx, qy, qz])
def getTestCamera(width):
    camera_lists = []

    znear = 100
    zfar = 0.01
    height = width / 2
    fovy = math.pi/2
    fovx = math.pi/2

    #相机位姿四元数qvec0和偏移向量T
    qvec0 = angle_axis_to_quaternion([30,7.8,0])
    T = np.array([0,0,-1.5])

    R0 = qvec2rotmat(qvec0)
    R = np.transpose(R0)
    camera_list = SphereCam(width, height, fovx, fovy, znear, zfar, R, T)
    camera_lists.append(camera_list)

    return camera_lists

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_test")
    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_sphere(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, width : int, train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, getTestCamera(width), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-width', '--width', type=int, default=2000, help='the width of image ')
    parser.add_argument("--train", action="store_false",default=True)
    parser.add_argument("--skip_test", action="store_true")
    parser.set_defaults(data_device = 'cuda')
    parser.set_defaults(model_path = 'D:\\repetition\\3dgs\\gaussian-splatting\\model\\playroom')
    parser.set_defaults(source_path = 'D:\\repetition\\3dgs\\gaussian-splatting\\data_playroom')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.width, args.train, args.skip_test)

