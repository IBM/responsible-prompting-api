FROM python:3.12
ENV PYTHONUNBUFFERED=1

COPY control /opt/microservices/control
COPY prompt-sentences-main /opt/microservices/prompt-sentences-main
COPY static /opt/microservices/static
COPY requirements.txt /opt/microservices/
COPY .env /opt/microservices/
COPY app.py /opt/microservices/
COPY config.py /opt/microservices/
COPY helpers /opt/microservices/helpers
COPY models /opt/microservices/models

RUN pip install --upgrade pip \
	&& pip install --upgrade pipenv\
	&& apt-get clean \
	&& apt-get update \
	&& apt install -y build-essential \
	&& apt install -y libmariadb3 libmariadb-dev \
	&& pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
	&& pip install --no-cache-dir --upgrade -r /opt/microservices/requirements.txt

USER 1001
EXPOSE 8080
WORKDIR /opt/microservices/
CMD ["python", "app.py", "8080"]
