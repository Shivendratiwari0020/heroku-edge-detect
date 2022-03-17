FROM python:3.6.9
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#FROM django:onbuild
# Install dependencies required for psycopg2 python package
#RUN apk update && apk add libpq
#RUN apk update && apk add --virtual .build-deps gcc python3-dev musl-dev postgresql-dev

RUN  apt-get update \
    && apt-get install -y --no-install-recommends \	
        postgresql-client \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . .

#RUN mv wait-for /bin/wait-for
RUN pip install --upgrade pip
#RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
#COPY . .


# Remove dependencies only required for psycopg2 build
#RUN apk del .build-deps

EXPOSE 8000

CMD ["python3", "project/manage.py", "runserver", "0.0.0.0:8000"]
