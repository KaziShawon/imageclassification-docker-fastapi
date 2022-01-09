FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
COPY ./installs /app/installs
# Below three lines are done because torch is a very large file, and we want to avoid downloading it
# again and again
RUN pip install ./installs/numpy-1.22.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install ./installs/Pillow-9.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install ./installs/torch-1.10.1-cp38-cp38-manylinux1_x86_64.whl
RUN pip install ./installs/torchvision-0.11.2-cp38-cp38-manylinux1_x86_64.whl

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . /app
RUN ls /app
RUN rm -rf /app/installs
EXPOSE 5000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
