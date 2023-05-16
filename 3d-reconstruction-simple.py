
from glob import glob
import numpy as np
import cv2
import numpy as np
import open3d as o3d
import os

from myhitnet.hitnet import HitNet, ModelType, CameraConfig


class NeuralStereoMatcher:
    def __init__(self, baseline_m, focal_length_px):
        # Select model type
        # model_type = ModelType.middlebury
        # model_type = ModelType.flyingthings
        model_type = ModelType.eth3d

        if model_type == ModelType.middlebury:
            model_path = "myhitnet/models/middlebury_d400.pb"
        elif model_type == ModelType.flyingthings:
            model_path = "myhitnet/models/flyingthings_finalpass_xl.pb"
        elif model_type == ModelType.eth3d:
            model_path = "myhitnet/models/eth3d.pb"

        cam = CameraConfig(baseline_m, focal_length_px)
        self.hitnet_depth = HitNet(model_path, model_type, cam)

    def inference(self, left_img, right_img):
        return self.hitnet_depth(left_img, right_img)


class LcadCameraConfig:
    # class CameraConfig:
    width = 640
    height = 480
    fx = 0.764749 * width
    fy = 1.01966 * height
    cx = 0.505423 * width
    cy = 0.493814 * height
    baseline = 0.24004


# class OlhoDoDonoCameraConfig:
class OddCameraConfig:
    fx = 696.475
    fy = 696.455
    cx = 637.755
    cy = 336.5585
    baseline = 0.120153


def stereo_matching_opencv(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    nchannels = 1
    blocksize = 7
    n_disparities = 64

    # para descrição dos parâmetros, https://docs.opencv.org/4.5.2/d2/d85/classcv_1_1StereoSGBM.html
    # veja tambem https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        # numDisparities=128,
        numDisparities=n_disparities,
        P1=8 * nchannels * blocksize * blocksize,
        P2=32 * nchannels * blocksize * blocksize,
        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        blockSize=blocksize,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,)

    # arguments are first left, then right: https://docs.opencv.org/4.5.2/d9/d07/classcv_1_1stereo_1_1StereoMatcher.html#:~:text=virtual%20void%20cv%3A%3Astereo%3A%3AStereoMatcher%3A%3Acompute,)
    # disparity is given for the left image: https://docs.opencv.org/4.5.2/dd/d53/tutorial_py_depthmap.html
    disparity = stereo.compute(grayL, grayR)

    # statistics about the disparity
    print(np.max(disparity), np.min(disparity), np.mean(disparity))
    # see https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp
    # Como a disparidade eh representada usando uint16, eles multiplicam por 16 para
    # manter um nivel razoavel de precisao razoavel dada a quantidade pequena de bits.
    disparity = disparity.astype(np.float) / 16.0

    return disparity


def reproject_3d_carmen(xl, yl, disparity, cam_config):
    # double xr = right_point.x - camera->xc;
    xl = xl - cam_config.cx
    # double yr = right_point.y - camera->yc;
    yl = yl - cam_config.cy
    # double xl = xr + disparity;
    xr = xl - disparity
    # double fx_fy = camera->fx / camera->fy;
    fx_fy = cam_config.fx / cam_config.fy
    #
    # double X = -(camera->baseline * camera->fx) / (xr - xl);
    Z = -(cam_config.baseline * cam_config.fx) / (xr - xl)
    # double Y = (camera->baseline * (xr + xl)) / (2.0 * (xr - xl));
    X = (cam_config.baseline * (xr + xl)) / (2.0 * (xr - xl))
    # double Z = -fx_fy * (camera->baseline * yr) / (xr - xl);
    Y = -fx_fy * (cam_config.baseline * yl) / (xr - xl)

    return X, Y, Z


def reproject_3d(xl, yl, disparity, cam_config):
    """
    # LIVRO Computer Vision: Algorithms and Applications (Szeliski), 2011, pag. 473
    # d = f * (B/Z), onde f eh a focal length em pixels, B é o baseline, d é a disparidade e Z é o depth
    # O link abaixo mostra como calcular X e Y:
    # https://stackoverflow.com/questions/41503561/whats-the-difference-between-reprojectimageto3dopencv-and-disparity-to-3d-coo
    #
    """
    #
    # Values in pixels
    #
    fx = cam_config.fx
    fy = cam_config.fy
    cx = cam_config.cx
    cy = cam_config.cy

    Z = (cam_config.baseline * fx) / disparity
    X = (xl - cx) * (Z / fx)
    Y = (yl - cy) * (Z / fy)

    return X, Y, Z


def view_disparity_img(left, right, disparity):
    # visualization of the disparity map
    d_view = np.copy(disparity)
    d_view -= np.min(d_view)
    d_view /= np.max(d_view)
    d_view *= 255
    d_view = d_view.astype(np.uint8)
    print("d_view.shape:", d_view.shape)
    d_view = cv2.cvtColor(d_view, cv2.COLOR_GRAY2BGR)
    print("d_view.shape:", d_view.shape)

    view = cv2.hconcat([left, right, d_view])

    mult = (1000 / view.shape[1])
    view = cv2.resize(view, (
        int(mult * view.shape[1]), 
        int(mult * view.shape[0])
    ))
    
    cv2.imshow("view", view)
    cv2.waitKey(1)


def view_point_cloud(left, disparity, cam_config):
    p3D = []

    for row in range(left.shape[0]):
        for column in range(left.shape[1]):
            d = disparity[row][column]

            # ignore small disparities
            if d < 1.5:
                continue

            # Nota: estou espelhando x e usando -d como entrada apenas
            # para a visualização ficar certa na lib que mostra a pointcloud
            # na tela.
            X, Y, Z = reproject_3d(
                left.shape[1] - column - 1, row, -d, cam_config)
            #X, Y, Z = reproject_3d_carmen(column, row, d)
            b, g, r = left[row][column]

            # ignore distant points for a better visualization
            if (X**2 + Y**2 + Z**2) ** 0.5 > 30.0:
                continue

            p3D.append([X, Y, Z, r, g, b])

    p3D = np.array(p3D)
    positions = p3D[:, :3]
    colors = p3D[:, 3:]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0, 0, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

images_left = []
images_right = []
def read_images_log():
    # Define a estrutura do registro do arquivo index

    path_txt = './logs_iara/logs_iara/log-volta-da-ufes-20181206.txt'
    path_root = './logs_iara/logs_iara'
    paths = []

    # abre o arquivo de dados para leitura
    lines = []
    with open(path_txt, 'r') as file:
        for line in file:
            if line.startswith('BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3'):
                lines.append(line.strip())

        data = []
        lines = sorted(lines, key=lambda x: float(x[-3]))
        for line in lines:
            path = line.split(" ")[1]
            path = path.replace("/dados", "")

            # new_path = path_root + path
            paths.append(path)
            # print(path)            

    ranger_init = 0
    ranger_end = 10        
    # Lê o arquivo de imagem
    for i in range(ranger_init, ranger_end):
        with open(path_root + paths[i], 'rb') as f:
            # Define o tamanho de cada imagem em bytes
            image_size = 640 * 480 * 3
            # Lê os dados das duas imagens
            data = f.read(image_size * 2)
            # Separa os dados das duas imagens
            data_left = data[:image_size]
            data_right = data[image_size:]
            # Converte os dados em arrays numpy
            image_left = np.frombuffer(data_left, dtype=np.uint8).reshape(480, 640, 3)
            image_right = np.frombuffer(data_right, dtype=np.uint8).reshape(480, 640, 3)
            images_left.append(image_left)
            images_right.append(image_right)

    print("Fim das leituras das imagens")
# # Exibe as imagens usando a função cv2_imshow do Google Colab
# cv2.imshow("left",image_left)
# cv2.imshow("right",image_right)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


def main():
    # left = cv2.imread("imgs/left6.png")
    # right = cv2.imread("imgs/right6.png")

    # Criar o diretório para salvar as disparidades, se não existir
    diretorio = "./disparidades/"
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

    read_images_log()
    for i in range(len(images_left)):
        left = images_left[i]
        right = images_right[i]
        cam_config = LcadCameraConfig
        # cam_config = OddCameraConfig

        neural_stereo = NeuralStereoMatcher(
            cam_config.baseline, cam_config.fx)

        # disparity = stereo_matching_opencv(left, right)
        disparity = neural_stereo.inference(left, right)

        # view_disparity_img(left, right, disparity)
        # view_point_cloud(left, disparity, cam_config)

        # print(disparity.dtype)

        # salvando disparidades
        # cv2.imwrite("./disparity.bin", disparity)
        disparity.tofile(f"./disparidades/disparidade{i}.bin")

if __name__ == "__main__":
    main()
