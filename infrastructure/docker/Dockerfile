FROM spark:3.5.6-scala2.12-java17-python3-ubuntu

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get -y install wget && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt/spark/jars && \
    wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar && \
    wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.4/hadoop-common-3.3.4.jar && \
    wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-mapreduce-client-core/3.3.4/hadoop-mapreduce-client-core-3.3.4.jar && \
    wget https://repo1.maven.org/maven2/org/wildfly/openssl/wildfly-openssl/1.0.7.Final/wildfly-openssl-1.0.7.Final.jar && \
    wget https://jdbc.postgresql.org/download/postgresql-42.7.4.jar

RUN mkdir -p /home/spark/work && \
    mkdir -p /home/spark/.vscode-server && \
    mkdir -p /home/spark/.local && \
    mkdir -p /home/spark/.cache

RUN chown -R spark:spark /home/spark

USER spark
WORKDIR /home/spark/work

ENV VIRTUAL_ENV=/home/spark/venv

COPY requirements.txt .
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache -r requirements.txt


ENV PATH="$VIRTUAL_ENV/bin:$SPARK_HOME/bin:${PATH}"
ENV PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip"

CMD [ "sleep", "infinity" ]