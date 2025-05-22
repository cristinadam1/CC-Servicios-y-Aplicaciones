FROM docker.io/bitnami/spark:3.3.0

# Cambio a root para poder instalar cosas
USER root

# InstaloPython 3 y pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalo numpy 
RUN pip3 install numpy

# Vuelvo a usuario por defecto de Spark
USER 1001
