FROM python:3.11.9
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

RUN mkdir ~/.streamlit

RUN cp config.toml ~/.streamlit/config.toml
ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]