FROM tensorflow/tensorflow:latest-gpu

COPY . yolo
WORKDIR yolo

RUN apt update
RUN apt install -y libgl1-mesa-glx libglib2.0-0

RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN pip3 install -r requirements-gpu.txt

RUN python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4

CMD ["python", "waitress_server.py"]

EXPOSE 5000