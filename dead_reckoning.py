import struct
import gps

# Define a estrutura do registro do arquivo index
index_format = "<8sL"
x_points = gps.x_points
y_points = gps.y_points

path = "../files"
# Abre o arquivo de dados para leitura
with open(path + "/arquivo.txt", "r") as data_file:
    # Lê o arquivo de índice
    with open(path + "/arquivo.index", "rb") as index_file:
        # Lê o primeiro registro do arquivo index, que contém o número de registros no arquivo
        count_bytes = index_file.read(struct.calcsize(index_format))
        count_record = struct.unpack(index_format, count_bytes)
        count = count_record[1]
        print(f"Número de registros: {count}")
        robot_lines = []
        # Lê cada registro do arquivo index e verifica se a linha correspondente no arquivo de dados contém "ROBOTVELOCITY_ACK"
        for i in range(count):
            index_bytes = index_file.read(struct.calcsize(index_format))
            if len(index_bytes) != struct.calcsize(index_format):
                break
            index_record = struct.unpack(index_format, index_bytes)
            offset = index_record[1]

            # Lê a linha correspondente ao registro atual no arquivo de dados
            data_file.seek(offset)
            line = data_file.readline().strip()

            # Verifica se a linha contém "ROBOTVELOCITY_ACK" e a armazena, se sim
            if "ROBOTVELOCITY_ACK" in line:
                robot_lines.append(line)

velocity = []
angle = []
for line in robot_lines:
    words = line.split(" ")
    velocity.append(words[1])
    angle.append(words[2])


# Convertendo elementos de velocity em float
for i in range(len(velocity)):
    velocity[i] = float(velocity[i])

# Convertendo elementos de angle em float
for i in range(len(angle)):
    angle[i] = float(angle[i])

# Verificando o tipo dos elementos após a conversão
print(type(velocity[0]))  # <class 'float'>
print(type(angle[0]))  # <class 'float'>

import math
import matplotlib.pyplot as plt

# Definindo as variáveis do modelo de Arckermann
L = 2.5  # Distância entre eixos do veículo
dt = 0.1  # Intervalo de tempo entre as medições
t = 0.0  # Tempo inicial
x = x_points[0]  # Posição inicial em x
y = y_points[0]  # Posição inicial em y
theta = math.radians(angle[0])  # Ângulo inicial em radianos
v = velocity[0]  # Velocidade inicial
beta = 0.0  # Ângulo de direção inicial

# Definindo as listas para armazenar os pontos de trajetória calculados
x_traj = [x]
y_traj = [y]

# x_traj = [0]
# y_traj = [0]

# Calculando a trajetória usando o modelo de Arckermann
for i in range(1, len(x_points)):
    # Calculando o ângulo de direção beta
    delta_x = x_points[i] - x
    delta_y = y_points[i] - y
    delta_theta = math.atan2(delta_y, delta_x) - theta
    delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
    beta = math.atan2(2.0 * L * math.sin(delta_theta), v * dt)
    
    # Atualizando as variáveis de estado do veículo
    theta = theta + (v / L) * math.tan(beta) * dt
    x = x + v * math.cos(theta) * dt
    y = y + v * math.sin(theta) * dt
    t = t + dt
    v = velocity[i]
    
    # Salvando os pontos de trajetória calculados
    x_traj.append(x)
    y_traj.append(y)


plt.plot(x_points, y_points, 'b-', label='Pontos')
plt.plot(x_traj, y_traj, 'r--', label='Dead Reckoning')
plt.legend()
plt.title('Dead Reckoning usando modelo de Arckermann')
plt.xlabel('Posição em X (m)')
plt.ylabel('Posição em Y (m)')
plt.show()

