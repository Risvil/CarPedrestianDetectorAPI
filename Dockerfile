FROM python:3.6.4

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' carpedestrianapi-user

WORKDIR /opt/carpedestrian_mrcnn

ENV FLASK_APP run.py

# Install requirements
ADD . /opt/carpedestrian_mrcnn
RUN pip install --upgrade pip
RUN pip install -r /opt/carpedestrian_mrcnn/requirements.txt
RUN pip install pycocotools

RUN chmod +x /opt/carpedestrian_mrcnn/samples/carpedrestian/car_pedrestian_api/run.sh
RUN chown -R carpedestrianapi-user ./

USER carpedestrianapi-user

EXPOSE 5000

WORKDIR /opt/carpedestrian_mrcnn/samples/carpedrestian/car_pedrestian_api
CMD ["bash", "./run.sh"]