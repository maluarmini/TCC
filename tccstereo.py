import matplotlib
import matplotlib.pyplot as plt
import struct
import math
import open3d as o3d
import numpy as np
import cv2
from myhitnet.hitnet import HitNet, ModelType, CameraConfig
import os

# define a estrutura do registro do arquivo index
index_format = "<8sL"
path_txt = './logs_iara/logs_iara/log-volta-da-ufes-20181206.txt'
path_index = './logs_iara/logs_iara/log-volta-da-ufes-20181206.txt.index'

matplotlib.use('TkAgg') 

# Velodyne vertical angles e ray order
velodyne_vertical_angles = [
    -30.6700000, -29.3300000, -28.0000000, -26.6700000, -25.3300000,
    -24.0000000, -22.6700000, -21.3300000, -20.0000000, -18.6700000,
    -17.3300000, -16.0000000, -14.6700000, -13.3300000, -12.0000000,
    -10.6700000, -9.3299999, -8.0000000, -6.6700001, -5.3299999,
    -4.0000000, -2.6700001, -1.3300000, 0.0000000, 1.3300000, 2.6700001,
    4.0000000, 5.3299999, 6.6700001, 8.0000000, 9.3299999, 10.6700000]
velodyne_ray_order = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9,
    11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

ranger_init = 1000
ranger_end = 1050

def carmen_global_convert_degmin_to_double(dm_format: float) -> float:
    degree = math.floor(dm_format / 100.0)
    minutes = (dm_format - degree * 100.0) / 60.0
    return degree + minutes

class GdcToUtm:
    RADIANS_PER_DEGREE = 0.0174532925199432957692
    PI = 4.0 * math.atan(1.0)
    CScale = .9996
    init = False

    A = 6378137
    F = 298.257223563

    # Create the ERM constants.
    F = 1 / (F)
    C = A * \
        (1 - F)
    Eps2 = F * \
        (2.0 - F)
    Eps25 = .25 * (Eps2)
    Epps2 = (
        Eps2) / (1.0 - Eps2)
    polx2b = 1.0 * Eps2 + \
        1.0 / 4.0 * math.pow(Eps2, 2) + 15.0/128.0 * \
        math.pow(Eps2, 3) - \
        455.0/4096.0 * math.pow(Eps2, 4)

    polx2b = 3.0/8.0 * polx2b

    polx3b = 1.0 * pow(Eps2, 2) + 3.0/4.0 * \
        math.pow(Eps2, 3) - 77.0 / \
        128.0 * math.pow(Eps2, 4)

    polx3b = 15.0 / 256.0 * polx3b

    polx4b = pow(Eps2, 3) - 41.0 / \
        32.0 * pow(Eps2, 4)

    polx4b = polx4b * 35.0 / 3072.0

    polx5b = -315.0 / \
        131072.0 * math.pow(Eps2, 4)

    poly1b = 1.0 - \
        (1.0/4.0 * Eps2) - (3.0/64.0 * math.pow(Eps2, 2)) - \
        (5.0/256.0 * math.pow(Eps2, 3)) - \
        (175.0/16384.0 * math.pow(Eps2, 4))

    poly2b = polx2b * -2.0 + polx3b * 4.0 - polx4b * \
        6.0 + polx5b * 8.0

    poly3b = polx3b * -8.0 + polx4b * \
        32.0 - polx5b * 80.0

    poly4b = polx4b * - \
        32.0 + polx5b * 192.0

    poly5b = polx5b * -128.0

    @staticmethod
    def Convert(latitude: float, longitude: float, elevation: float, north_south, east_west):
        latitude = carmen_global_convert_degmin_to_double(float(latitude))
        longitude = carmen_global_convert_degmin_to_double(float(longitude))

        # verify the latitude and longitude orientations
        if ('S' == north_south):
            latitude = -latitude
        if ('W' == east_west):
            longitude = -longitude

        utm_z = elevation

        if (latitude < 0):
            utm_hemisphere_north = False
        else:
            utm_hemisphere_north = True

        # if (gdc.longitude < 0.0) // XXX - reddy, 11 Sep 98
        # gdc.longitude += 360.0

        source_lat = latitude * GdcToUtm.RADIANS_PER_DEGREE
        source_lon = longitude * GdcToUtm.RADIANS_PER_DEGREE

        s1 = math.sin(source_lat)
        c1 = math.cos(source_lat)
        tx = s1 / c1
        s12 = s1 * s1

        # USE IN-LINE SQUARE ROOT
        rn = GdcToUtm.A / ((.25 - GdcToUtm.Eps25*s12 + .9999944354799/4) +
                           (.25-GdcToUtm.Eps25*s12)/(.25 - GdcToUtm.Eps25*s12 + .9999944354799/4))

        # COMPUTE UTM COORDINATES

        # Compute Zone
        utm_zone = int(source_lon * 30.0 / GdcToUtm.PI + 31)

        if (utm_zone <= 0):
            utm_zone = 1
        else:
            if (utm_zone >= 61):
                utm_zone = 60

        axlon0 = (utm_zone * 6 - 183) * GdcToUtm.RADIANS_PER_DEGREE

        al = (source_lon - axlon0) * c1

        sm = s1 * c1 * (GdcToUtm.poly2b + s12 * (GdcToUtm.poly3b + s12 *
                                                 (GdcToUtm.poly4b + s12 * GdcToUtm.poly5b)))

        sm = GdcToUtm.A * \
            (GdcToUtm.poly1b * source_lat + sm)

        tn2 = tx * tx
        cee = GdcToUtm.Epps2 * c1 * c1
        al2 = al * al
        poly1 = 1.0 - tn2 + cee
        poly2 = 5.0 + tn2 * (tn2 - 18.0) + cee * (14.0 - tn2 * 58.0)

        # COMPUTE EASTING
        utm_x = GdcToUtm.CScale * rn * al * \
            (1.0 + al2 * (.166666666666667 * poly1 + .00833333333333333 * al2 * poly2))

        utm_x += 5.0E5

        # COMPUTE NORTHING

        poly1 = 5.0 - tn2 + cee * (cee * 4.0 + 9.0)
        poly2 = 61.0 + tn2 * (tn2 - 58.0) + cee * (270.0 - tn2 * 330.0)

        utm_y = GdcToUtm.CScale * (sm + rn * tx * al2 * (0.5 + al2 *
                                                         (.0416666666666667 * poly1 + .00138888888888888 * al2 * poly2)))

        if (source_lat < 0.0):
            utm_y += 1.0E7

        return utm_x, utm_y, utm_z, utm_zone, utm_hemisphere_north

class GPS:
    def read_and_ordination_gps():  
        lines = []
        with open(path_txt, 'r') as file:
            for line in file:
                if line.startswith('NMEAGGA 1'):
                    lines.append(line.strip())

            lines = sorted(lines, key=lambda x: float(x.split()[-3]))
            return lines   
    
    def calcular_angulos(x_point, y_point,i, time,points_angle, x_points, y_points):

        p0 = [x_point, y_point]
        theta = 0
        for i in range(len(x_points)):
            if ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5 > 5:
                theta = math.atan2(y_points[i], x_points[i])
                # faz uma tupla com o angulo e o ponto
                points_angle.append((x_point, y_point, theta, time))
                break

        return points_angle

    def gps_conversation(data, x_points, y_points):
        convert_data = []
        for line in data:
            x, y, z, _, _ = GdcToUtm.Convert(
                line[0], line[1], line[2], line[3], line[4])
            convert_data.append([x, y, z])

        for ponto in convert_data:
            x, y, _ = ponto
            x_points.append(x - convert_data[0][0])
            y_points.append(y - convert_data[0][1])

        return x_points, y_points

def gps():
    x_points = []
    y_points = []
    times_gps = []
    points_angle = []
    print("INIT GPS\n")
    data = []
    lines = GPS.read_and_ordination_gps()
    for line in lines:
        words = line.split(" ")
        data.append([words[3], words[5], words[11], words[4], words[6]])            
        times_gps.append(float(words[-3]))
    
    x_points, y_points = GPS.gps_conversation(data, x_points, y_points)

    for i in range(len(x_points)):
        points_angle = GPS.calcular_angulos(x_points[i], y_points[i],i, times_gps[i],points_angle, x_points, y_points)
    print("END GPS\n") 
    return points_angle, x_points, y_points

# Variaveis GPS
points_angle_gps = []
x_points = []
y_points = []
points_angle_gps, x_points, y_points = gps() 

print(points_angle_gps[0])
print(points_angle_gps[-1])
print(x_points[0])
print(x_points[-1])
print(y_points[0])
print(y_points[-1])
class DeadReckoning:

    def read_and_ordination_dead_reckoning():
        lines = []
        with open(path_txt, 'r') as file:
            for line in file:
                if line.startswith('ROBOTVELOCITY_ACK'):
                    lines.append(line.strip())

            lines = sorted(lines, key=lambda x: float(x.split()[-3]))
            return lines  

    def convert_velocity_and_angle(velocity, angle):
        v_m = 1.0
        a_m = 0.89
        a_add = -0.004

         # Convertendo elementos de velocity e angle em float
        for i in range(len(velocity)):
            velocity[i] = float(velocity[i]) * v_m
            angle[i] = float(angle[i]) * a_m + a_add
        return velocity, angle

    def calculate_initial_angle(x_points,y_points):
        p0 = [x_points[0], y_points[0]]
        for i in range(len(x_points)):
            if ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5 > 5:
                theta = math.atan2(y_points[i] - p0[1], x_points[i] - p0[0])
                break
        
        return theta

    def arckemann_model(velocity, angle, times_dead_reckoning, theta):
        points_angle_dead_reckoning = []
        # Definindo as variáveis do modelo de Arckermann

        L = 2.625  # Distância entre eixos do veículo
        dt = 0.1  # Intervalo de tempo entre as medições
        t = 0.0  # Tempo inicial
        x = x_points[0]  # Posição inicial em x
        y = y_points[0]  # Posição inicial em y
        v = velocity[0]  # Velocidade inicial

        # Definindo as listas para armazenar os pontos de trajetória calculados
        x_traj = [x]
        y_traj = [y]

        # Calculando a trajetória usando o modelo de Arckermann
        for i in range(1, len(velocity)):

            # Atualizando as variáveis de estado do veículo
            dt = times_dead_reckoning[i] - times_dead_reckoning[i - 1]
            v = velocity[i]
            x = x + v * math.cos(theta) * dt
            y = y + v * math.sin(theta) * dt
            theta = theta + (v / L) * math.tan(angle[i]) * dt
            time = times_dead_reckoning[i]
            # Salvando os pontos de trajetória calculados
            x_traj.append(x)
            y_traj.append(y)

            points_angle_dead_reckoning.append((x, y, theta, time))

        return points_angle_dead_reckoning

def dead_reckoning(x_points,y_points):
    velocity = []
    angle = []
    times_dead_reckoning = []
    points_angle_dead_reckoning = []
    print("Init Dead reckoning") 
    lines = DeadReckoning.read_and_ordination_dead_reckoning()
    for line in lines:
        words = line.split(" ")
        velocity.append(words[1])
        angle.append(words[2])
        times_dead_reckoning.append(float(words[-3]))  

    velocity, angle = DeadReckoning.convert_velocity_and_angle(velocity, angle)  
    theta = DeadReckoning.calculate_initial_angle(x_points,y_points)
    points_angle_dead_reckoning = DeadReckoning.arckemann_model(velocity, angle, times_dead_reckoning, theta)
    print("End Dead reckoning")
    return points_angle_dead_reckoning

#Variaveis dead reckoning
points_angle_dead_reckoning = []
points_angle_dead_reckoning = dead_reckoning(x_points, y_points)

print(points_angle_dead_reckoning[0])
print(points_angle_dead_reckoning[-1])

class Velodyne:
    def read_and_ordination_dead_reckoning():
        lines = []
        with open(path_txt, 'r') as file:
            for line in file:
                if line.startswith('VELODYNE_PARTIAL_SCAN_IN_FILE'):
                    lines.append(line.strip())

            lines = sorted(lines, key=lambda x: float(x.split()[-3]))
            return lines  

path_array = []
number_shots_array = []
times_clouds_points = []
data = []
clouds_points = []

def velodyne_data():
    # Velodyne
    print("INIT VELODYNE")
    # Abre o arquivo de dados para leitura
    points_velodyne = []
    lines = []
    lines = Velodyne.read_and_ordination_dead_reckoning()
    for line in lines:
        path = line.split(" ")[1]
        path1 = path.replace("/dados", "")
        path1 = "./logs_iara/logs_iara/" + path1
        path_array.append(path1)

        number_shots = line.split(" ")[2]
        number_shots_array.append(int(number_shots))

        # path da nuvem de pontos, o number_shot respectivo e o timeestamp
        points_velodyne.append([path1, int(number_shots), float(line.split(" ")[-3])])

                    
    # for i in range(len(path_array)):
    for j in range(ranger_init, ranger_end):
        data = []
        points = []
        points_time = []
        colors = []

        with open(points_velodyne[j][0], "rb") as f:
            
            for i in range(points_velodyne[j][1]):
                laser_data = {}
                laser_data['h_angle'] = struct.unpack('d', f.read(8))[0]
                laser_data['points'] = []

                ranges = struct.unpack('32h', f.read(64))

                reflectances = struct.unpack('32B', f.read(32))

                for k in range(32):
                    laser_data['points'].append((
                        ranges[velodyne_ray_order[k]] / 500.0,
                        velodyne_vertical_angles[k],
                        reflectances[velodyne_ray_order[k]],
                        points_velodyne[j][2]
                    ))
                data.append(laser_data)

        
        for shot in data:
            horiz_angle = -shot['h_angle']
            horiz_angle = np.radians(horiz_angle)
            for dist, vert_angle, reflect, time in shot['points']:
                vert_angle = np.radians(vert_angle)

                if (dist * np.cos(vert_angle)) < 5.0 :
                    continue

                x = dist * np.cos(vert_angle) * np.cos(horiz_angle)
                y = dist * np.cos(vert_angle) * np.sin(horiz_angle)
                z = dist * np.sin(vert_angle)
                reflect *= np.clip((5 / 255.0), 0, 1)
                points.append([x, y, z])
                points_time.append([x, y, z, time])
                colors.append([reflect, reflect, reflect])

        # Criar uma nuvem de pontos Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        clouds_points.append(pcd)
        times_clouds_points.append(points_velodyne[j][-1])

        pcd.colors = o3d.utility.Vector3dVector(colors)
    print("END VELODYNE\n")
    
velodyne_data()

print(clouds_points[0], times_clouds_points[0])
print(clouds_points[-1], times_clouds_points[-1])

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

            p3D.append([X, Y, Z, b, g, r])

    p3D = np.array(p3D)
    positions = p3D[:, :3]
    colors = p3D[:, 3:]

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.get_render_option().background_color = np.asarray([0, 0, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()

    return pcd

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
    
    # Lê o arquivo de imagem
    # for i in range(len(paths)):
    for i in range(ranger_init,ranger_end):
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

def read_disparity():
    folder_path = "./disparidades"  # Caminho para a pasta "disparity" na raiz

    # Lista todos os arquivos na pasta "disparity"
    file_list = os.listdir(folder_path)

    # Inicializa um array vazio para armazenar os dados binários
    disparity = []

    # Percorre cada arquivo na pasta
    for file_name in file_list[:10]:
        file_path = os.path.join(folder_path, file_name)

        # Abre o arquivo binário para leitura
        with open(file_path, "rb") as file:
            # Lê os dados binários do arquivo e os armazena no array "disparity"
            data = file.read()
            disparity.append(data)

    
    return disparity


def read_disparity_files(directory, num_files):
    disparities = []
    for i in range(num_files):
        file_path = os.path.join(directory, f"disparidade{i}.bin")
        disparity = np.fromfile(file_path, dtype=np.float32)
        disparity = disparity.reshape((480, 640))  # Substitua 'height' e 'width' pelos tamanhos corretos das imagens
        disparities.append(disparity)
    return disparities


clouds_points=[]
disparity = []
'''
def main():
    disparity = read_disparity_files()
    # Converte a lista de dados binários em um array numpy
    # disparity = np.array(disparity)
    read_images_log()
    for i in range(ranger_init,ranger_end):
        left = images_left[i]
        right = images_right[i]
        cam_config = LcadCameraConfig
        # cam_config = OddCameraConfig

        # neural_stereo = NeuralStereoMatcher(
        #     cam_config.baseline, cam_config.fx)

        # disparity = stereo_matching_opencv(left, right)
        # disparity = neural_stereo.inference(left, right)

        # view_disparity_img(left, right, disparity)
        clouds_points.append(view_point_cloud(left, disparity[i], cam_config))


'''

def main():
    # Defina o diretório onde as disparidades foram salvas
    directory = "./disparidades/"
    num_files = 50  # Número total de arquivos de disparidade

    # Lê as disparidades dos arquivos binários
    disparities = read_disparity_files(directory, num_files)
    read_images_log()
    for i in range(len(disparities)):
        disparity = disparities[i]
        left = images_left[i]
        right =  images_right[i]
        cam_config = LcadCameraConfig
        # cam_config = OddCameraConfig

        # view_disparity_img(left, right, disparity)
        clouds_points.append(view_point_cloud(left, disparity, cam_config))

        # print(disparity.dtype)

        # Disparidades estão prontas para serem usadas

if __name__ == "__main__":
    main()
print('end program')

def transform_point_cloud(pcd, pose):
    # Matriz de rotação
    x,y,theta,time = pose
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

    # Vetor de translação
    t = np.array([x, y, 0])

    # Aplicar rotação e translação na nuvem de pontos
    pcd.rotate(R)
    pcd.translate(t)
    return pcd


points_angle = points_angle_dead_reckoning

pcd_list = []

for i, (cloud_point, timestamp) in enumerate(zip(clouds_points, times_clouds_points)):
    
    closest_idx = np.argmin(np.abs([p[3] - timestamp for p in points_angle]))

    transformed_pcd = transform_point_cloud(cloud_point, points_angle[closest_idx])

    pcd_list.append(transformed_pcd)



# Definir a cor de fundo da visualização como azul
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
for pcd in pcd_list:
    visualizer.add_geometry(pcd)
visualizer.get_render_option().background_color = np.asarray([0.1, 0.1, 0.9])

# Visualizar a nuvem de pontos com o background azul
visualizer.run()
visualizer.destroy_window()

