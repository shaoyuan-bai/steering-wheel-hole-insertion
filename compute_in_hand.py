# coding=utf-8

"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os
import logging

import  yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from libs.auxiliary import find_latest_data_folder
from libs.log_setting import CommonLog

from save_poses import poses_main

np.set_printoptions(precision=8,suppress=True)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)


current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"eye_hand_data")

images_path = os.path.join("eye_hand_data",find_latest_data_folder(current_path))
file_path = os.path.join(images_path,"poses.txt")  #采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应


with open("config.yaml", 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

XX = data.get("checkerboard_args").get("XX") #标定板的中长度对应的角点的个数
YY = data.get("checkerboard_args").get("YY") #标定板的中宽度对应的角点的个数
L = data.get("checkerboard_args").get("L")   #标定板一格的长度  单位为米
hand_eye_args = data.get("hand_eye_args", {})
MAX_REPROJECTION_ERROR_PX = float(hand_eye_args.get("max_reprojection_error_px", 1.5))


def calc_reprojection_errors(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    errors = []
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
        errors.append(float(err))
    return errors


def run_hand_eye_methods(r_tool, t_tool, rvecs, tvecs):
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    results = {}
    for name, method in methods.items():
        try:
            rotation_matrix, translation_vector = cv2.calibrateHandEye(
                r_tool,
                t_tool,
                rvecs,
                tvecs,
                method=method,
            )
            results[name] = (rotation_matrix, translation_vector)
            logger_.info(f"{name} hand-eye rotation:\n{rotation_matrix}")
            logger_.info(f"{name} hand-eye translation:\n{translation_vector}")
        except Exception as exc:
            logger_.warning(f"{name} hand-eye failed: {exc}")
    return results


def func():

    path = os.path.dirname(__file__)

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点
    image_indices = []

    images_num = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

    for i in range(1, len(images_num) + 1):   #标定好的图片在images_path路径下，从0.jpg到x.jpg

        image_file = os.path.join(images_path,f"{i}.jpg")

        if os.path.exists(image_file):

            logger_.info(f'读 {image_file}')

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:

                obj_points.append(objp)
                image_indices.append(i)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)
    if N < 4:
        raise RuntimeError(f"有效棋盘格图片不足: {N}")

    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    reproj_errors = calc_reprojection_errors(obj_points, img_points, rvecs, tvecs, mtx, dist)
    keep = [idx for idx, err in enumerate(reproj_errors) if err <= MAX_REPROJECTION_ERROR_PX]
    for image_index, err in zip(image_indices, reproj_errors):
        logger_.info(f"image {image_index}.jpg reprojection_error_px={err:.4f}")

    if len(keep) < N:
        logger_.warning(
            f"过滤重投影误差 > {MAX_REPROJECTION_ERROR_PX:.3f}px 的图片: "
            f"{N - len(keep)}/{N}"
        )
    if len(keep) < 4:
        raise RuntimeError(f"过滤后有效棋盘格图片不足: {len(keep)}")

    if len(keep) != N:
        obj_points = [obj_points[i] for i in keep]
        img_points = [img_points[i] for i in keep]
        image_indices = [image_indices[i] for i in keep]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        reproj_errors = calc_reprojection_errors(obj_points, img_points, rvecs, tvecs, mtx, dist)
        N = len(img_points)

    # logger_.info(f"内参矩阵:\n:{mtx}" ) # 内参数矩阵
    # logger_.info(f"畸变系数:\n:{dist}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    logger_.info(f"平均重投影误差(px): {float(np.mean(reproj_errors)):.4f}")
    logger_.info(f"最大重投影误差(px): {float(np.max(reproj_errors)):.4f}")

    print("-----------------------------------------------------")

    poses_main(file_path)
    # 机器人末端在基座标系下的位姿

    csv_file = os.path.join(path,"RobotToolPose.csv")
    tool_pose = np.loadtxt(csv_file,delimiter=',')

    R_tool = []
    t_tool = []

    for image_index in image_indices:
        pose_idx = image_index - 1
        R_tool.append(tool_pose[0:3, 4 * pose_idx:4 * pose_idx + 3])
        t_tool.append(tool_pose[0:3, 4 * pose_idx + 3])

    results = run_hand_eye_methods(R_tool, t_tool, rvecs, tvecs)
    if "TSAI" not in results:
        raise RuntimeError("TSAI hand-eye calibration failed.")

    rotation_matrix, translation_vector = results["TSAI"]
    return rotation_matrix, translation_vector

if __name__ == '__main__':

    # 旋转矩阵
    rotation_matrix, translation_vector = func()

    # 将旋转矩阵转换为四元数
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    x, y, z = translation_vector.flatten()

    logger_.info(f"旋转矩阵是:\n {            rotation_matrix}")

    logger_.info(f"平移向量是:\n {            translation_vector}")

    logger_.info(f"四元数是：\n {             quaternion}")
