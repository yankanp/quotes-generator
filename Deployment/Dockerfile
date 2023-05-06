FROM --platform=linux/x86_64 python:3.10.7
MAINTAINER Priyankan Kiru "pkirupaharan@uri.edu"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["quotesflask.py"]