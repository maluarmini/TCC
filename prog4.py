import struct
import open3d as o3d
import numpy as np


path1 = '/log-volta-da-ufes-20181206.txt_velodyne-20230501T182408Z-001/log-volta-da-ufes-20181206.txt_velodyne/1544120000/1544127500/1544127566.780115.pointcloud'

path2 = '/log-volta-da-ufes-20181206.txt_velodyne-20230501T182408Z-001/log-volta-da-ufes-20181206.txt_velodyne/1544120000/1544127500/1544127566.830139.pointcloud'

path = './arquivo.pointcloud'
num_shots = 1084

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


data = []
with open(path, "rb") as f:
    for i in range(num_shots):
        laser_data = []
        laser_data.append(struct.unpack('d', f.read(8))[0])

        ranges = struct.unpack('32h', f.read(64))
        
        reflectances = struct.unpack('32B', f.read(32))
        
        for j in range(32):
            laser_data.append(ranges[velodyne_ray_order[j]] / 500.0)
            laser_data.append(velodyne_vertical_angles[j])
            laser_data.append(reflectances[velodyne_ray_order[j]])
        data.append(laser_data)

points = []
for i in range(num_shots):
    for j in range(32):
        dist = data[i][j*3+1]
        horiz_angle = np.radians((j % 16) * 22.5)
        # horiz_angle = laser_data[j+32]
        vert_angle = np.radians(velodyne_vertical_angles[j])
        x = dist * np.cos(vert_angle) * np.sin(horiz_angle)
        y = dist * np.cos(vert_angle) * np.cos(horiz_angle)
        z = dist * np.sin(vert_angle)
        points.append([x, y, z])

# Criar uma nuvem de pontos Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualizar a nuvem de pontos
o3d.visualization.draw_geometries([pcd])