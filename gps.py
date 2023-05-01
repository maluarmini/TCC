# Converter lat e long para x,y,z
import math


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
    

import struct

# define a estrutura do registro do arquivo index
index_format = "<8sL"
path = "../files"

# abre o arquivo de dados para leitura
with open(path + "/arquivo.txt", "r") as data_file:
    # lê o arquivo de índice
    with open(path + "/arquivo.index", "rb") as index_file:
        # lê o primeiro registro do arquivo index, que contém o número de registros no arquivo
        count_bytes = index_file.read(struct.calcsize(index_format))
        count_record = struct.unpack(index_format, count_bytes)
        count = count_record[1]
        print(f"Número de registros: {count}")
        nmea_lines = []
        # lê cada registro do arquivo index e verifica se a linha correspondente no arquivo de dados começa com "NMEAGGA"
        for i in range(count):
            index_bytes = index_file.read(struct.calcsize(index_format))
            if len(index_bytes) != struct.calcsize(index_format):
                break
            index_record = struct.unpack(index_format, index_bytes)
            offset = index_record[1]

            # lê a linha correspondente ao registro atual no arquivo de dados
            data_file.seek(offset)
            line = data_file.readline().strip()

            # verifica se a linha começa com "NMEAGGA 1" e a armazena, se sim
            if line.startswith("NMEAGGA 1"):
                nmea_lines.append(line)

        data = []
        # imprime o terceiro elemento de cada linha NMEA
        for line in nmea_lines:
            words = line.split(" ")
            # print(words[3] + " " + words[5] + " " + words[11] + " " + words[4] + " " + words[6])
            data.append([words[3], words[5], words[11], words[4], words[6]])


convert_data = []
for line in data:
    x, y, z, _, _ = GdcToUtm.Convert(line[0], line[1], line[2], line[3], line[4])
    convert_data.append([x, y, z])


# for i in convert_data:
#     print(i)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x_points = []
y_points = []

for ponto in convert_data:
    x, y, _ = ponto
    x_points.append(x)
    y_points.append(y)

plt.scatter(x_points, y_points)
plt.show()