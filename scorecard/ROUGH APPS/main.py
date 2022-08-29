from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def read_main():
    return ("Message": "My first API APP")
@app.get("/article/{article_id}")