import os
import io

from fastapi import templating
from classifcation import Classification
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile, applications
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

model = Classification()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/files", StaticFiles(directory="files"), name="files")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    print(request)
    content = file.file.read()
    saved_filepath = f'files/{file.filename}'
    with open(saved_filepath, 'wb') as f:
        f.write(content)
    output = model.predict_from_path(saved_filepath)
    payload = {'request': request,
               "filename": file.filename,
               'output': output}
    return templates.TemplateResponse("home.html", payload)


@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB')
    image = image.resize((224, 224), Image.ANTIALIAS)
    output = model.predict_from_image(image)
    return output
