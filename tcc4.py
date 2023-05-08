import matplotlib
import matplotlib.pyplot as plt
import struct
import math
import open3d as o3d
import numpy as np

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

# define a estrutura do registro do arquivo index
index_format = "<8sL"

path_txt = './logs_iara/logs_iara/log-volta-da-ufes-20181206.txt'
path_index = './logs_iara/logs_iara/log-volta-da-ufes-20181206.txt.index'

points_angle = []

def calcular_angulos(x_point, y_point):
    p0 = [x_point, y_point]
    theta = 0
    for i in range(len(x_points)):
        dist = ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5
        if ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5 > 5:
            theta = math.atan2(y_points[i] - p0[1], x_points[i] - p0[0])
            # faz uma tupla com o angulo e o ponto
            points_angle.append((x_point, y_point, theta))
            break

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


convert_data = []
x_points = []
y_points = []
times_gps = []

def gps_data():
    # GPS
    print("GPS INIT")
    # abre o arquivo de dados para leitura
    lines = []
    with open(path_txt, 'r') as file:
        for line in file:
            if line.startswith('NMEAGGA 1'):
                lines.append(line.strip())

        data = []
        for line in lines:
            words = line.split(" ")
            data.append([words[3], words[5], words[11], words[4], words[6]])            
            times_gps.append(float(words[-1]))

    for line in data:
        x, y, z, _, _ = GdcToUtm.Convert(
            line[0], line[1], line[2], line[3], line[4])
        convert_data.append([x, y, z])

    matplotlib.use('TkAgg') 

    for ponto in convert_data:
        x, y, _ = ponto
        x_points.append(x)
        y_points.append(y)
    
    # plt.scatter(x_points, y_points)
    # plt.show()        
    for i in range(len(x_points)):
        calcular_angulos(x_points[i], y_points[i])

gps_data()
print("END GPS\n") 


velocity = []
angle = []
times_dead_reckoning = []
points_angle_dead_reckoning = []

def dead_reckoning_data():
    # DEAD RECKONING
    print("DEAD RECKONING INIT")
    # abre o arquivo de dados para leitura
    lines = []
    with open(path_txt, 'r') as file:
        for line in file:
            if line.startswith('ROBOTVELOCITY_ACK'):
                lines.append(line.strip())
    
        for line in lines:
            words = line.split(" ")
            velocity.append(words[1])
            angle.append(words[2])
            times_dead_reckoning.append(float(words[-1]))

    v_m = 1.0
    a_m = 0.89
    a_add = -0.004

    # Convertendo elementos de velocity em float
    for i in range(len(velocity)):
        velocity[i] = float(velocity[i]) * v_m

    # Convertendo elementos de angle em float
    for i in range(len(angle)):
        angle[i] = float(angle[i]) * a_m + a_add


    # Angulo inicial
    p0 = [x_points[0], y_points[0]]
    for i in range(len(x_points)):
        if ((p0[0] - x_points[i]) ** 2 + (p0[1] - y_points[i]) ** 2) ** 0.5 > 5:
            theta = math.atan2(y_points[i] - p0[1], x_points[i] - p0[0])
            break

            # Definindo as variáveis do modelo de Arckermann
    L = 2.5  # Distância entre eixos do veículo
    dt = 0.1  # Intervalo de tempo entre as medições
    t = 0.0  # Tempo inicial
    x = x_points[0]  # Posição inicial em x
    y = y_points[0]  # Posição inicial em y
    v = velocity[0]  # Velocidade inicial
    beta = 0.0  # Ângulo de direção inicial

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

        # Salvando os pontos de trajetória calculados
        x_traj.append(x)
        y_traj.append(y)

        points_angle_dead_reckoning.append((x, y, theta))
    
dead_reckoning_data()
print("END DEAD RECKONING\n")

path_array = []
number_shots_array = []
times_clouds_points = []
data = []
clouds_points = []
points = []
colors = []

ranger_init = 0
ranger_end = 2
def velodyne_data():
    # Velodyne
    print("INIT VELODYNE")
    # Abre o arquivo de dados para leitura
    lines = []
    with open(path_txt, 'r') as file:
        for line in file:
            if line.startswith('VELODYNE_PARTIAL_SCAN_IN_FILE'):
                lines.append(line.strip())

        for line in lines:
            path = line.split(" ")[1]
            path1 = path.replace("/dados", "")
            path1 = "./logs_iara/logs_iara/" + path1
            path_array.append(path1)

            number_shots = line.split(" ")[2]
            number_shots_array.append(int(number_shots))

            times_clouds_points.append(float(line.split(" ")[-1]))
                    
    
    # for i in range(len(path_array)):
    for j in range(ranger_init, ranger_end):
        with open(path_array[j], "rb") as f:
            for i in range(number_shots_array[j]):
                laser_data = {}
                laser_data['h_angle'] = struct.unpack('d', f.read(8))[0]
                laser_data['points'] = []

                ranges = struct.unpack('32h', f.read(64))

                reflectances = struct.unpack('32B', f.read(32))

                for j in range(32):
                    laser_data['points'].append((
                        ranges[velodyne_ray_order[j]] / 500.0,
                        velodyne_vertical_angles[j],
                        reflectances[velodyne_ray_order[j]]
                    ))
                data.append(laser_data)

        
        for shot in data:
            horiz_angle = shot['h_angle']
            horiz_angle = np.radians(horiz_angle)
            for dist, vert_angle, reflect in shot['points']:
                vert_angle = np.radians(vert_angle)
                x = dist * np.cos(vert_angle) * np.sin(horiz_angle)
                y = dist * np.cos(vert_angle) * np.cos(horiz_angle)
                z = dist * np.sin(vert_angle)
                reflect *= np.clip((10 / 255.0), 0, 1)
                points.append([x, y, z])
                colors.append([reflect, reflect, reflect])

        # Criar uma nuvem de pontos Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        clouds_points.append(pcd)
        pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualizar a nuvem de pontos
# o3d.visualization.draw_geometries([clouds_points[0]])
    
velodyne_data()
print("END VELODYNE\n")

'''
def transform_point_cloud(pcd, pose):
    # Extrair a rotação e translação da pose
    x, y, theta = pose

    # Calcular a matriz de rotação 2D
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Criar a matriz de transformação 4x4
    transform = np.eye(4)
    transform[:2, :2] = rotation[:2, :2]
    transform[:2, 3] = [x, y]

    
    # Aplicar a transformação na nuvem de pontos
    transformed_pcd = pcd.transform(transform)

    return transformed_pcd
'''

def transform_point_cloud(pcd, pose):
    # Matriz de rotação
    x,y,theta = pose
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

    # Vetor de translação
    t = np.array([x, y, 0])

    # Aplicar rotação e translação na nuvem de pontos
    pcd.rotate(R)
    pcd.translate(t)
    return pcd


# points_angle = points_angle_dead_reckoning

pcd_list = []

for i, cloud_point in enumerate(clouds_points):
    # if velocity[i+ranger_init] < 0.2:
    #     continue

    # closest_idx = find_nearest_timestamp(times_gps, times_clouds_points[i+ranger_init])
    closest_idx = np.argmin(np.abs(np.array(times_gps) - times_clouds_points[i+ranger_init]))

    # Transformar a nuvem de pontos
    transformed_pcd = transform_point_cloud(cloud_point, points_angle[closest_idx])
    transformed_pcd.points = o3d.utility.Vector3dVector(np.asarray(transformed_pcd.points))

    # Adicionar a nuvem de pontos transformada à lista
    pcd_list.append(transformed_pcd)
    break

# o3d.visualization.draw_geometries(pcd_list)

# Definir a cor de fundo da visualização como azul
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
for pcd in pcd_list:
    visualizer.add_geometry(pcd)
# visualizer.add_geometry(pcd_list)
visualizer.get_render_option().background_color = np.asarray([0.1, 0.1, 0.9])

# Visualizar a nuvem de pontos com o background azul
visualizer.run()
visualizer.destroy_window()
