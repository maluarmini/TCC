from dataclasses import dataclass
import math
import matplotlib
import matplotlib.pyplot as plt
import struct
import math
import open3d as o3d
import numpy as np
import cv2
from myhitnet.hitnet import HitNet, ModelType, CameraConfig
import os

# Path principal do arquivo de dados
LOG_PATH = './logs_iara/logs_iara/log-volta-da-ufes-20181206.txt'

# Path a ser retirado para ficar apenas com o path do arquivo de nuvem de pontos
INITIAL_PATH = "./logs_iara/logs_iara/" 

# Path onde estão salvos os binários das disparidades
PATH_DISPARITYS = "./disparidades/"

RANGE_INIT = 2000
RANGE_END = 2100
# define a estrutura do registro do arquivo index
index_format = "<8sL"

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

@dataclass
class GpsMsg:
    latitude: float
    longitude: float
    altitude: float
    north_south: float
    east_west: float
    timestamp: float
    
    @staticmethod
    def parse(line):
        words = line.split(" ")
        return GpsMsg(float(words[3]), float(words[5]), float(words[11]), words[4], words[6], float(words[-3]))

class GPS:
    @staticmethod
    def read_and_ordination_gps():
        print("Init GPS")  
        lines = []
        with open(LOG_PATH, 'r') as file:
            for line in file:
                if line.startswith('NMEAGGA 1'):
                    lines.append(line.strip())

            lines = sorted(lines, key=lambda x: float(x.split()[-3]))

            return [GpsMsg.parse(line) for line in lines]

class GpsFormatter:
    @staticmethod
    def calcular_angulos(x_point, y_point,i, time,points_angle, x_points, y_points):
        p0 = [x_point, y_point]
        theta = 0
        for i in range(len(x_points)):
            if ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5 > 5:
                theta = math.atan2(y_points[i], x_points[i])
                # faz uma tupla com o angulo, ponto, timestamp
                points_angle.append((x_point, y_point, theta, time))
                break

        return points_angle

    @staticmethod
    def gps_conversation(gps_messages):
        x_points = []
        y_points = []
        times_gps = []
        points_angle = []
        convert_data = []
        for gps_msg in gps_messages:
            x, y, z, _, _ = GdcToUtm.Convert(
                gps_msg.latitude, gps_msg.longitude, gps_msg.altitude,  gps_msg.north_south,  gps_msg.east_west)
            convert_data.append([x, y, z])

        for ponto in convert_data:
            x, y, _ = ponto
            x_points.append(x - convert_data[0][0])
            y_points.append(y - convert_data[0][1])
            times_gps.append(gps_msg.timestamp)

        for i in range(len(x_points)):
            points_angle = GpsFormatter.calcular_angulos(x_points[i], y_points[i], i, times_gps[i], points_angle, x_points, y_points)
        print("End GPS") 
        return points_angle, x_points, y_points

@dataclass
class OdometryMsg:
    speed: float
    steering_angle: float
    timestamp: float

    @staticmethod
    def parse(line):
        words = line.split(" ")
        return OdometryMsg(float(words[1]), float(words[2]), float(words[-3]))

class DeadReckoning:
    v_m = 1.0
    a_m = 0.89
    a_add = -0.004
    L = 2.625  # Distância entre eixos do veículo
    

    @staticmethod
    def read_and_ordination_dead_reckoning():
        print("Init Dead Reckoning")
        lines = []
        with open(LOG_PATH, 'r') as file:
            for line in file:
                if line.startswith('ROBOTVELOCITY_ACK'):
                    lines.append(line.strip())

        lines = sorted(lines, key=lambda x: float(x.split()[-3]))

        return [OdometryMsg.parse(line) for line in lines]

    @staticmethod
    def convert_velocity_and_angle(odometry_msg):
        # Convertendo elementos de velocity e angle em float
        odometry_msg.speed = odometry_msg.speed * DeadReckoning.v_m
        odometry_msg.steering_angle = odometry_msg.steering_angle * DeadReckoning.a_m + DeadReckoning.a_add
        return odometry_msg

    @staticmethod
    def calculate_initial_angle(x_points,y_points):
        p0 = [x_points[0], y_points[0]]
        for i in range(len(x_points)):
            if ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5 > 5:
                theta = math.atan2(y_points[i] - p0[1], x_points[i] - p0[0])
                break
        
        return theta

    @staticmethod
    def arckemann_model(odometry_messages, x_points, y_points):
        points_angle_dead_reckoning = []
        # Tempo inicial
        t = 0.0
        # Posição inicial em x e y
        x = x_points[0]
        y = y_points[0]

        # Converter velocidade e angulo
        for i in range(len(odometry_messages)):
            odometry_messages[i] = DeadReckoning.convert_velocity_and_angle(odometry_messages[i])

        # Velocidade e ângulo inicial
        v = odometry_messages[0].speed
        theta = DeadReckoning.calculate_initial_angle(x_points, y_points)

        for i, odometry_msg in enumerate(odometry_messages, 1):
            if i == len(odometry_messages):
                break
            dt = odometry_messages[i].timestamp - odometry_messages[i - 1].timestamp
            v = odometry_msg.speed
            a = odometry_msg.steering_angle
            t = odometry_msg.timestamp

            x += v * dt * math.cos(theta)
            y += v * dt * math.sin(theta)
            theta += (v/DeadReckoning.L) * math.tan(a) * dt

            points_angle_dead_reckoning.append((x, y, theta, t))

        print("End Dead Reckoning")
        return points_angle_dead_reckoning

@dataclass
class VelodyneCloud:
    timestamp: float
    num_shots: int
    path_shots_data: str
    shots_data: list = None  # Inicializado como None

    @staticmethod
    def parse(line, INITIAL_PATH):
        words = line.split(" ")
        path = words[1].replace("/dados", "")
        path_shots_data = INITIAL_PATH + path
        num_shots = int(words[2])
        timestamp = float(words[-3])

        return VelodyneCloud(timestamp, num_shots, path_shots_data)

class Velodyne:
    @staticmethod
    def read_and_ordination_velodyne():
        print("Init Velodyne")
        lines = []
        with open(LOG_PATH, 'r') as file:
            for line in file:
                if line.startswith('VELODYNE_PARTIAL_SCAN_IN_FILE'):
                    lines.append(line.strip())

        lines = sorted(lines, key=lambda x: float(x.split()[-3]))

        return [VelodyneCloud.parse(line,  INITIAL_PATH) for line in lines]
    
    @staticmethod
    def velodyne_clouds(velodyne_clouds):
        # for i,cloud in enumerate(velodyne_clouds):
        for j in range(RANGE_INIT, RANGE_END):
            # if (i < ranger_init) or (i > ranger_end):
            #     continue
            velodyne_clouds[j].shots_data = []  # Inicialize shots_data como uma lista vazia
            with open(velodyne_clouds[j].path_shots_data, "rb") as f:
                for _ in range(velodyne_clouds[j].num_shots):
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
                            velodyne_clouds[j].timestamp
                        ))
                    velodyne_clouds[j].shots_data.append(laser_data)

    @staticmethod
    def create_points_clouds(velodyne_clouds):
        Velodyne.velodyne_clouds(velodyne_clouds)  # Preencher os dados dos tiros
        clouds_points = []
        times_clouds_points = []
        
        for cloud in velodyne_clouds:
            if cloud.shots_data is None:  
                continue
            points, colors = [], []
            for shot in cloud.shots_data:  # cloud.shots_data já é uma lista de dados de tiro
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
                    colors.append([reflect, reflect, reflect])

            # Create a point cloud in Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            clouds_points.append(pcd)
            times_clouds_points.append(cloud.timestamp)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        print("End Velodyne\n")
        return clouds_points, times_clouds_points

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

def create_pcd_list(clouds_points, times_clouds_points, points_angle):
    pcd_list = []

    for i, (cloud_point, timestamp) in enumerate(zip(clouds_points, times_clouds_points)):
        
        closest_idx = np.argmin(np.abs([p[3] - timestamp for p in points_angle]))

        transformed_pcd = transform_point_cloud(cloud_point, points_angle[closest_idx])

        pcd_list.append(transformed_pcd)
    
    return pcd_list

def view_map(pcd_list):
    # Definir a cor de fundo da visualização como azul
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    for pcd in pcd_list:
        visualizer.add_geometry(pcd)
    visualizer.get_render_option().background_color = np.asarray([0.1, 0.1, 0.9])

    # Visualizar a nuvem de pontos com o background azul
    visualizer.run()
    visualizer.destroy_window()

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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd


@dataclass
class StereoImage:
    timestamp: float
    path_image_data: str
    image_left: np.array = None
    image_right: np.array = None

    @staticmethod
    def parse(line):
        words = line.split(" ")
        path = words[1].replace("/dados", "")
        path_image_data = INITIAL_PATH + path
        timestamp = float(words[-3])

        return StereoImage(timestamp, path_image_data)

class Bumblebee:
    @staticmethod
    def read_images_log():
        print("Reading Images")
        lines = []
        with open(LOG_PATH, 'r') as file:
            for line in file:
                if line.startswith('BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3'):
                    lines.append(line.strip())

        lines = sorted(lines, key=lambda x: float(x[-3]))

        return [StereoImage.parse(line) for line in lines]

    @staticmethod
    def load_images(images):
        image_size = 640 * 480 * 3
        for i in range(RANGE_INIT, RANGE_END):
            with open(images[i].path_image_data, 'rb') as f:
                data = f.read(image_size * 2)
                data_left = data[:image_size]
                data_right = data[image_size:]
                images[i].image_left = np.frombuffer(data_left, dtype=np.uint8).reshape(480, 640, 3)
                images[i].image_right = np.frombuffer(data_right, dtype=np.uint8).reshape(480, 640, 3)
        print("End reading images")

    def read_disparity_files(directory):
        disparities = []
        for i in range(RANGE_INIT, RANGE_END):
            file_path = os.path.join(directory, f"disparidade{i}.bin")
            disparity = np.fromfile(file_path, dtype=np.float32)
            disparity = disparity.reshape((480, 640))  # Substitua 'height' e 'width' pelos tamanhos corretos das imagens
            disparities.append(disparity)
        return disparities
    
    @staticmethod
    def create_point_clouds(images, disparities, cam_config):
        Bumblebee.load_images(images)
        clouds_points = []
        for i in range(len(disparities)):
            disparity = disparities[i]
            left = images[i].image_left
            clouds_points.append(view_point_cloud(left, disparity, cam_config))
        return clouds_points

def main():
    gps_messages = GPS.read_and_ordination_gps()
    points_angle_gps, x_points, y_points = GpsFormatter.gps_conversation(gps_messages)

    dead_reckoning_messages = DeadReckoning.read_and_ordination_dead_reckoning()
    points_angle_dead_reckoning = DeadReckoning.arckemann_model(dead_reckoning_messages,x_points, y_points)

    velodyne_clouds_messages = Velodyne.read_and_ordination_velodyne()  # Obter as nuvens de Velodyne
    clouds_points, times_clouds_points = Velodyne.create_points_clouds(velodyne_clouds_messages)
    
    images = Bumblebee.read_images_log()
    cam_config = LcadCameraConfig
    # clouds_points = []
    # clouds_points = Bumblebee.create_point_clouds(images, Bumblebee.read_disparity_files(PATH_DISPARITYS), cam_config)

    pcd_list = create_pcd_list(clouds_points, times_clouds_points, points_angle_dead_reckoning)
    view_map(pcd_list)
    
if __name__ == "__main__":
    main()