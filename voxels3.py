
import copy
from tqdm import tqdm
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
INITIAL_PATH = "./logs_iara/logs_iara"

# Path onde estão salvos os binários das disparidades
PATH_DISPARITYS = "./disparidades/"

# painel da ufes (com quebra mola)
# RANGE_INIT = 2770
# RANGE_END = 2830

# inicio do log em movimento
RANGE_INIT = 0
RANGE_END = 5

# centro de linguas
# RANGE_INIT = 1275
# RANGE_END = 1325

# RANGE_INIT = 0
# RANGE_END = 1

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


def load_params(log):
    param_lines = [line for line in log if line.startswith('PARAM')]
    params = {}
    for line in param_lines:
        line = line.split()
        params[line[1]] = "".join(line[2:-3])
    return params


class Transforms:
    def __init__(self, params):
        # TODO: considerar colocar a rotacao que rotaciona o sistema de coordenadas
        # da imagem aqui tambem.
        self.vel2board = Transforms._transformation_from_params(
            params['velodyne_x'],
            params['velodyne_y'],
            params['velodyne_z'],
            params['velodyne_roll'],
            params['velodyne_pitch'],
            params['velodyne_yaw']
        )

        self.cam2board = Transforms._transformation_from_params(
            params['camera3_x'],
            params['camera3_y'],
            params['camera3_z'],
            params['camera3_roll'],
            params['camera3_pitch'],
            params['camera3_yaw']
        )

        self.gps2board = Transforms._transformation_from_params(
            params['gps_nmea_x'],
            params['gps_nmea_y'],
            params['gps_nmea_z'],
            params['gps_nmea_roll'],
            params['gps_nmea_pitch'],
            params['gps_nmea_yaw']
        )

        self.board2car = Transforms._transformation_from_params(
            params['sensor_board_1_x'],
            params['sensor_board_1_y'],
            params['sensor_board_1_z'],
            params['sensor_board_1_roll'],
            params['sensor_board_1_pitch'],
            params['sensor_board_1_yaw']
        )

        self.vel2car = np.dot(self.board2car, self.vel2board)
        self.cam2car = np.dot(self.board2car, self.cam2board)
        self.gps2car = np.dot(self.board2car, self.gps2board)
        self.car2world = np.eye(4)
        self._update_transforms_to_world()

    def update_pose(self, pose):
        self.car2world = Transforms._transformation_from_params(
            pose[0], pose[1], 0, 0, 0, pose[2])
        self._update_transforms_to_world()

    def _update_transforms_to_world(self):
        self.vel2world = np.dot(self.car2world, self.vel2car)
        self.cam2world = np.dot(self.car2world, self.cam2car)
        self.gps2world = np.dot(self.car2world, self.gps2car)

    @staticmethod
    def _transformation_from_params(x, y, z, roll, pitch, yaw):
        T = np.eye(4)
        T[:3, :3] = \
            o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))
        T[:3, 3] = [x, y, z]
        return T


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
    def calcular_angulos(x_point, y_point, i, time, points_angle, x_points, y_points):
        p0 = [x_point, y_point]
        theta = 0
        for j in range(i, len(x_points)):
            if ((p0[0] - x_points[j]) ** 2 + (p0[1] - y_points[j]) ** 2) ** 0.5 > 5:
                theta = math.atan2(y_points[j] - p0[1], x_points[j] - p0[0])
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
            convert_data.append([x, y, z, gps_msg.timestamp])

        for ponto in convert_data:
            x, y, _, timestamp = ponto
            x_points.append(x - convert_data[0][0])
            y_points.append(y - convert_data[0][1])
            times_gps.append(timestamp)

        for i in range(len(x_points)):
            points_angle = GpsFormatter.calcular_angulos(
                x_points[i], y_points[i], i, times_gps[i], points_angle, x_points, y_points)
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
        odometry_msg.steering_angle = odometry_msg.steering_angle * \
            DeadReckoning.a_m + DeadReckoning.a_add
        return odometry_msg

    @staticmethod
    def calculate_initial_angle(x_points, y_points):
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
            odometry_messages[i] = DeadReckoning.convert_velocity_and_angle(
                odometry_messages[i])

        # Velocidade e ângulo inicial
        v = odometry_messages[0].speed
        theta = DeadReckoning.calculate_initial_angle(x_points, y_points)

        for i, odometry_msg in enumerate(odometry_messages, 1):
            if i == len(odometry_messages):
                break
            dt = odometry_messages[i].timestamp - \
                odometry_messages[i - 1].timestamp
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
            # Inicialize shots_data como uma lista vazia
            velodyne_clouds[j].shots_data = []
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
        # Preencher os dados dos tiros
        Velodyne.velodyne_clouds(velodyne_clouds)
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

                    if (dist * np.cos(vert_angle)) < 5.0:
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

        print("End Velodyne")
        return clouds_points, times_clouds_points


def transform_point_cloud(pcd, pose):
    # Matriz de rotação
    x, y, theta, time = pose
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    # Vetor de translação
    t = np.array([x, y, 0])

    # Aplicar rotação e translação na nuvem de pontos
    pcd.rotate(R, center=[0, 0, 0])
    pcd.translate(t)
    return pcd


def draw_car_coordinate_system(pose):
    (x, y, th, _) = pose
    # x = y = th = 0
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axes = axes.translate((x, y, 0))
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, th))
    axes = axes.rotate(R)
    return axes

VOXEL_SIZE = 0.2
WIDTH = 20
LENGTH = 20
HEIGHT = 5

# Classe bloco
class Bloco:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.voxels = None

    def criar_bloco(x, y, z):
        bloco = Bloco(x, y, z)
        tamanho_voxel = VOXEL_SIZE
        tamanho_matriz = (int(LENGTH / tamanho_voxel), int(WIDTH / tamanho_voxel), int(HEIGHT / tamanho_voxel))
        matriz_voxels = np.empty((tamanho_matriz[0], tamanho_matriz[1], tamanho_matriz[2]), dtype=object)

        for x in range(tamanho_matriz[0]):
            for y in range(tamanho_matriz[1]):
                for z in range(tamanho_matriz[2]):
                    matriz_voxels[x, y, z] = Voxel(0, 0, 0)

        print("matriz_voxels criada e inicializada - novo bloco criado")
        bloco.voxels = matriz_voxels
        return bloco

    @staticmethod
    def salvar_bloco(bloco, arquivo):
        bloco_data = np.array(bloco, dtype=object)
        bloco_data.tofile(arquivo)

    @staticmethod
    def carregar_bloco(arquivo):
        bloco_data = np.fromfile(arquivo, dtype=object)
        bloco = np.reshape(bloco_data, (3, 3))
        return bloco

class Voxel:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.color_x = x
        self.color_y = y
        self.color_z = z
        self.quantidade_pontos = 0

    def adicionar_ponto(self, ponto, cor):
        self.x += ponto[0]
        self.y += ponto[1]
        self.z += ponto[2]
        self.color_x += cor[0]
        self.color_y += cor[1]
        self.color_z += cor[2]
        self.quantidade_pontos += 1

tamanho_voxel = VOXEL_SIZE
tamanho_matriz = (int(LENGTH / 0.2),int(WIDTH/0.2),int(HEIGHT/0.2))


def adicionar_ponto_na_matriz(bloco, ponto, color):
    x, y, z = ponto
    voxel_x = abs(x // tamanho_voxel)
    voxel_y = abs(y // tamanho_voxel)
    voxel_z = abs(z // tamanho_voxel)

    matriz = bloco.voxels

    if ((voxel_x >=0 and voxel_x < tamanho_matriz[0]) and (voxel_y >=0 and voxel_y < tamanho_matriz[1]) and (voxel_z >=0 and voxel_z < tamanho_matriz[2])):
        matriz[int(voxel_x)][int(voxel_y)][int(voxel_z)].adicionar_ponto(ponto, color)


# lista de blocos do mapa
list_blocos = []
def verificar_pose_em_bloco(x_pose, y_pose, z_pose, x_min, x_max, y_min, y_max, z_min, z_max):
    if x_pose >= x_min and x_pose <= x_max and y_pose >= y_min and y_pose <= y_max and z_pose >= z_min and z_pose <= z_max:
        return True
    else:
        return False

def conferir_bloco_memoria(bloco_x, bloco_y, bloco_z):
    for bloco in memoria:
        if bloco.x == bloco_x and bloco.y == bloco_y and bloco.z == bloco_z:
            return True  # O bloco está na memória

    return False  # O bloco não está na memória

def conferir_bloco_hd(x, y, z):
    nome_arquivo = f"{x}_{y}_{z}.bin"
    caminho_hd = os.path.join(os.getcwd(), "hd")
    caminho_arquivo = os.path.join(caminho_hd, nome_arquivo) 
    return os.path.isfile(caminho_arquivo)

def obter_bloco_memoria(x, y, z):
    for bloco in memoria:
        if bloco.x == x and bloco.y == y and bloco.z == z:
            return bloco
    return None

def adicionar_bloco_memoria(bloco):
    memoria.append(bloco)

def verificar_adjacencia(bloco1, bloco2):
    """Verifica se os dois blocos são adjacentes."""
    x1, y1, z1 = bloco1.x, bloco1.y, bloco1.z
    x2, y2, z2 = bloco2.x, bloco2.y, bloco2.z

    return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)) <= 1

def salvar_bloco_hd(bloco):
    """Salva o bloco no HD."""
    arquivo_bloco_hd = f"{bloco.x}_{bloco.y}_{bloco.z}.bin"
    bloco.salvar_bloco(arquivo_bloco_hd)

memoria = []
def create_pcd_list(clouds_points, times_clouds_points, points_angle, tf: Transforms, sensor: str):
    pcd_list = []
    # memoria = []
    bloco = Bloco.criar_bloco(0, 0, 0)
    adicionar_bloco_memoria(bloco)

    bloco_x_anterior = 0
    bloco_y_anterior = 0
    bloco_z_anterior = 0 

    for i, (cloud_point, timestamp) in enumerate(zip(clouds_points, times_clouds_points)):
        closest_idx = np.argmin(np.abs([p[3] - timestamp for p in points_angle]))
        tf.update_pose(points_angle[closest_idx])

        x, y, z, _ = points_angle[i]
        bloco_x = abs(x // LENGTH)
        bloco_y = abs(y // WIDTH)
        bloco_z = abs(z // HEIGHT)
        
        # Verifica se o bloco atual é diferente do anterior
        if not (bloco_x == bloco_x_anterior and bloco_y == bloco_y_anterior and bloco_z == bloco_z_anterior):
            
            # Se for diferente, verifica se e a minha memória principal ainda tem memória disponível
            # Confiro qual o bloco mais distante da minha posição atual
            # Salvo este bloco no HD e libero memória na memória principal
            if len(memoria) >= (WIDTH * LENGTH):
                bloco_mais_distante = None
                maior_distancia = -math.inf
                for b in memoria:
                    distancia = math.sqrt((bloco_x - b.x)**2 + (bloco_y - b.y)**2 + (bloco_z - b.z)**2)
                    if distancia > maior_distancia:
                        maior_distancia = distancia
                        bloco_mais_distante = b

                if bloco_mais_distante and not verificar_adjacencia(bloco_mais_distante, bloco):
                    salvar_bloco_hd(bloco_mais_distante)
                    memoria.remove(bloco_mais_distante)

            # Verifica se o bloco atual está na memória principal
            if conferir_bloco_memoria(bloco_x, bloco_y, bloco_z):
                bloco = obter_bloco_memoria(bloco_x, bloco_y, bloco_z)
            
            # Verifica se o bloco atual está no HD
            elif conferir_bloco_hd(bloco_x, bloco_y, bloco_z):
                arquivo_bloco_hd = f"{bloco_x}_{bloco_y}_{bloco_z}.bin"
                bloco = Bloco.carregar_bloco(arquivo_bloco_hd)
                adicionar_bloco_memoria(bloco)

            # Se não estiver em nenhum dos dois, cria um novo bloco
            else:
                bloco = Bloco.criar_bloco(bloco_x, bloco_y, bloco_z)
                adicionar_bloco_memoria(bloco)

            # O meu bloco atual vira o novo bloco anterior
            bloco_x_anterior = bloco_x
            bloco_y_anterior = bloco_y
            bloco_z_anterior = bloco_z

        if sensor == 'velodyne':
            transformed_pcd = cloud_point.transform(tf.vel2world)
        else:
            transformed_pcd = cloud_point.transform(tf.cam2world)

        for ponto, color in zip(transformed_pcd.points, transformed_pcd.colors):
            adicionar_ponto_na_matriz(bloco, ponto, color)

    pontos = []
    cores = []

    # Loop através de todos os blocos na memória
    for bloco in memoria:

        matriz = bloco.voxels
        for x in range(tamanho_matriz[0]):
            for y in range(tamanho_matriz[1]):
                for z in range(tamanho_matriz[2]):
                    voxel = matriz[x][y][z]
                    if (voxel.quantidade_pontos > 0):
                        pontos.append([voxel.x/voxel.quantidade_pontos,voxel.y/voxel.quantidade_pontos, voxel.z/voxel.quantidade_pontos])
                        cores.append([voxel.color_x/voxel.quantidade_pontos, voxel.color_y/voxel.quantidade_pontos, voxel.color_z/voxel.quantidade_pontos])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pontos)
    pcd.colors = o3d.utility.Vector3dVector(cores)
    pcd_list.append(pcd)
    return pcd_list

def view_map(pcd_list):
    # Definir a cor de fundo da visualização como azul
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    for pcd in pcd_list:
        visualizer.add_geometry(pcd)
    visualizer.get_render_option().background_color = np.asarray([
        0.1, 0.1, 0.9])

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
        # 150 a 390
        left_crop = left_img[150:390, :, :]
        right_crop = right_img[150:390, :, :]

        disparity_croped = self.hitnet_depth(left_crop, right_crop)

        disparity = np.zeros([left_img.shape[0], left_img.shape[1]])
        disparity[150:390, :] = disparity_croped

        return disparity

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
    disparity = disparity.astype(np.float64) / 16.0

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
        # corta o ceu
        if row < 150:
            continue

        for column in range(left.shape[1]):
            # corta o carro
            # if (column > 150) and (row > 390):
            if row > 390:
                continue

            d = disparity[row][column]

            # ignore small disparities
            if d < 1.5:
                continue

            X, Y, Z = reproject_3d(
                column, row, d, cam_config)

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

        lines = sorted(lines, key=lambda x: float(x.split()[-3]))

        return [StereoImage.parse(line) for line in lines]

    @staticmethod
    def load_images(images):
        image_size = 640 * 480 * 3
        for i in range(RANGE_INIT, RANGE_END):
            with open(images[i].path_image_data, 'rb') as f:
                data = f.read(image_size * 2)
                data_left = data[:image_size]
                data_right = data[image_size:]
                images[i].image_left = np.frombuffer(
                    data_left, dtype=np.uint8).reshape(480, 640, 3)
                images[i].image_right = np.frombuffer(
                    data_right, dtype=np.uint8).reshape(480, 640, 3)
        print("End reading images")

    def read_disparity_files(directory):
        disparities = []

        for i in range(RANGE_INIT, RANGE_END):
            file_path = os.path.join(directory, f"disparidade{i}.bin")
            disparity = np.fromfile(file_path, dtype=np.float32)
            # Substitua 'height' e 'width' pelos tamanhos corretos das imagens
            disparity = disparity.reshape((480, 640))
            disparities.append(disparity)
        return disparities

    @staticmethod
    def create_point_clouds(images, disparities, cam_config):
        Bumblebee.load_images(images)
        clouds_points = []
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
        neural_stereo = NeuralStereoMatcher(
            cam_config.baseline, cam_config.fx)

        for i in tqdm(range(RANGE_INIT, RANGE_END)):
            # disparity = # disparities[i]
            left = images[i].image_left
            right = images[i].image_right

            left = np.clip(left.astype(np.float64) *
                           2, 0, 255).astype(np.uint8)
            right = np.clip(right.astype(np.float64) *
                            2, 0, 255).astype(np.uint8)

            disparity = neural_stereo.inference(left, right)
            # disparity = stereo_matching_opencv(left, right)

            view = np.copy(disparity)
            view -= np.min(view)
            view /= np.max(view)
            view *= 255
            view = view.astype(np.uint8)
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
            cv2.imshow("disparity", np.hstack(
                [cv2.cvtColor(left, cv2.COLOR_RGB2BGR), view]))
            cv2.waitKey(5)

            cloud = view_point_cloud(left, disparity, cam_config)
            cloud = cloud.rotate(R, center=[0, 0, 0])
            clouds_points.append(cloud)
        return clouds_points

def main():
    """
    Sistema de coordenadas do open3d: 
        X: vermelho 
        Y: verde
        Z: azul
    Sistemas de coordenadas padrão: 
        X: para frente
        Y: para esquerda
        Z: para cima
    Sistemas de coordenadas da câmera: 
        X: para direita
        Y: para baixo
        Z: para frente
    """

    with open(LOG_PATH, 'r') as file:
        log = file.readlines()

    params = load_params(log)

    tf = Transforms(params)

    gps_messages = GPS.read_and_ordination_gps()
    points_angle_gps, x_points, y_points = GpsFormatter.gps_conversation(
        gps_messages)

    dead_reckoning_messages = DeadReckoning.read_and_ordination_dead_reckoning()
    points_angle_dead_reckoning = DeadReckoning.arckemann_model(
        dead_reckoning_messages, x_points, y_points)

    # points_angle_dead_reckoning = points_angle_gps

    """
    # Obter as nuvens de Velodyne
    velodyne_clouds_messages = Velodyne.read_and_ordination_velodyne()
    clouds_points, times_clouds_points = Velodyne.create_points_clouds(
        velodyne_clouds_messages)
    pcd_list = create_pcd_list(
        clouds_points, times_clouds_points, points_angle_dead_reckoning, tf, 'velodyne')
    """
    pcd_list = []

    images = Bumblebee.read_images_log()
    cam_config = LcadCameraConfig
    clouds_points = []
    disparities = []
    # disparities = Bumblebee.read_disparity_files(PATH_DISPARITYS)
    clouds_points = Bumblebee.create_point_clouds(
        images, disparities, cam_config)
    times_clouds_points = [img.timestamp
                           for img in images[RANGE_INIT:RANGE_END]]
    # """
    pcd_list += create_pcd_list(
        clouds_points, times_clouds_points, points_angle_dead_reckoning, tf, 'camera')

    view_map(pcd_list)
    # """

if __name__ == "__main__":
    main()

