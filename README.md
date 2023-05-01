# TCC :closed_book:

Nesta primeira parte foi iniciado os trabalhos com alguns dos principais dados dos logs da IARA.

## Passos :pushpin:

- [Mensagens de interese do log](#instalação) <br>
> **XSENS_QUAT**: IMU <br>
> **VELODYNE_PARTIAL_SCAN_IN_FILE**: nuvem de pontos do velodyne <br>
> **ROBOTVELOCITY_ACK**: odometria <br>
> **NMEAGGA**: GPS. <br>
> **BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3**: imagens da câmera estéreo <br> <br>
- [Códigos](#uso)
> **gps.py** - plotar a curva gerada com os dados do gps <br>

> **dead_reckoning.py** plotar a curva estimada com o modelo de Ackermann <br>

> **image.py** visualizar uma imagem com o opencv <br> <br>

- [Para usar](#contribuição)

> Para usar os códigos é preciso além de ter os arquivos de log da IARA, definir o path_root para acessar os arquivos.
> Além disso, para estimar os valores no arquivo ``` dead_reckoning.py ``` é preciso anteriormente já ter gerados os pontos com o ``` gps.py ``` 

- [Execução](#execução)

> Para executar os arquivos - ``` python {nomedoarquivo.py} ```

## Requeriments :file_folder:

Esta é uma lista de requisitos necessários para o funcionamento correto do projeto.

- **python - 3.8.10**
- **matplotlib - 3.7.1**
- **opencv - 4.7.0.72**
- **open3D - 0.13.0**
- **struct - 1.5.4**
- **numpy - 1.24.3**



