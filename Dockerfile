# Dockerfile, Image, Container
FROM ubuntu:20.04

RUN apt update && apt -y upgrade
RUN apt install -y python3-pip

WORKDIR /opt/
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir .streamlit
COPY .streamlit/config.toml .streamlit/
COPY datamover.py .
COPY main.py .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
