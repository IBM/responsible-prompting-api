FROM python:3.9
ENV PYTHONUNBUFFERED=1

COPY control /opt/microservices/control
COPY prompt-sentences-main /opt/microservices/prompt-sentences-main
COPY static /opt/microservices/static
COPY requirements.txt /opt/microservices/
COPY .env /opt/microservices/
COPY app.py /opt/microservices/
COPY config.py /opt/microservices/
COPY helpers /opt/microservices/helpers

RUN pip install --upgrade pip \
	&& pip install --upgrade pipenv\
	&& apt-get clean \
	&& apt-get update \
	&& apt install -y build-essential \
	&& apt install -y libmariadb3 libmariadb-dev \
	&& pip install --upgrade -r /opt/microservices/requirements.txt

USER 1001
EXPOSE 8080
WORKDIR /opt/microservices/
CMD ["python", "app.py", "8080"]