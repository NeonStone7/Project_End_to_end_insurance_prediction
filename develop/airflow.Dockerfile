FROM apache/airflow:2.3.2
USER root
# Set timezone for image
RUN apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Saigon /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
ENV TZ="Asia/Saigon"
USER airflow
# Install Scrapy
RUN pip install scrapy psycopg2-binary mlflow scikit-learn nltk boto3