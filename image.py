import struct
import cv2
import numpy as np

# Define a estrutura do registro do arquivo index
index_format = "<8sL"
path_root = '../files'

# Abre o arquivo de dados para leitura
with open(path_root + "/arquivo.txt", "r") as data_file:
    # Lê o arquivo de índice
    with open(path_root + "/arquivo.index", "rb") as index_file:
        # Lê o primeiro registro do arquivo index, que contém o número de registros no arquivo
        count_bytes = index_file.read(struct.calcsize(index_format))
        count_record = struct.unpack(index_format, count_bytes)
        count = count_record[1]
        print(f"Número de registros: {count}")
        
        # Lê cada registro do arquivo index e verifica se a linha correspondente no arquivo de dados contém "BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3"
        for i in range(count):
            index_bytes = index_file.read(struct.calcsize(index_format))
            if len(index_bytes) != struct.calcsize(index_format):
                break
            index_record = struct.unpack(index_format, index_bytes)
            offset = index_record[1]

            # Lê a linha correspondente ao registro atual no arquivo de dados
            data_file.seek(offset)
            line = data_file.readline().strip()

            # Verifica se a linha contém "BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3" no começo da linha e extrai o caminho do arquivo
            if line.startswith("BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3"):
                path = line.split(" ")[1]
                path = path.replace("/dados", "")

                new_path = path_root + path
                print(new_path)
                # break  

# Lê o arquivo de imagem
with open(path_root + "/log-volta-da-ufes-20181206.txt_bumblebee/1544120000/1544127500/1544127566.907253.bb3.image", 'rb') as f:
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

# Exibe as imagens usando a função cv2_imshow do Google Colab
cv2.imshow("left",image_left)
cv2.imshow("right",image_right)

cv2.waitKey(0)
cv2.destroyAllWindows()
