import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fpdf import FPDF
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.middleware.cors import CORSMiddleware





# database

DATABASE_URL = "mysql+pymysql://root:ROOT@127.0.0.1:3306/ros2"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit= False, autoflush=False,bind=engine)


Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db

    finally:
        db.close()

class A_ItemModel(Base):
    __tablename__ ="items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    price = Column(Integer)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ItemModel(BaseModel):
    name: str
    price: int

app = FastAPI()

# CORS

app.add_middleware(
        CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)


# PDF
Pdf_folder = "pdf_folder"

os.makedirs(Pdf_folder,exist_ok=True)

class PDFGenerator:
    def __init__(self, file_name):
        self.filename = file_name
        self.pdf = FPDF()
    
    def create_pdf(self, content):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(200, 10, txt=content, ln=True, align="C")
        self.pdf.output(self.filename)

    def delete_pdf(self):
        os.remove(self.filename)
    
def get_pdf_generator(id:int):
    pdf_filename = os.path.join(Pdf_folder, f"document_{id}.pdf")
    return PDFGenerator(pdf_filename)



# hello world
@app.get("/")
def hello():
    return {"message":"hello world"}

# database crud
@app.get("/get_item/{id}", response_model=dict , tags=["DATABASE"])
def get_item(id:int, session: Session = Depends(get_db)):
    item = session.query(A_ItemModel).filter_by(id=id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item = item.to_dict()
    return {"data": item}

@app.post("/add_item" , tags=["DATABASE"])
def add_item(item: ItemModel, session: Session = Depends(get_db)):
    item = A_ItemModel(**dict(item))
    session.add(item)
    session.commit()
    item = item.to_dict()
    return{"data": item}

@app.delete("/delete_item/{id}" , tags=["DATABASE"])
def delete_item(id:int, session: Session = Depends(get_db)):
    item= session.query(A_ItemModel).filter_by(id=id).first()
    session.delete(item)
    session.commit()
    item = item.to_dict()

    return {"data": item}

@app.put("/update_item/{id}", tags=["DATABASE"])
def update_item(id: int, sesseion: Session = Depends(get_db)):
    item = sesseion.query(A_ItemModel).filter_by(id=id).first()
    item.price = item.price+1
    sesseion.commit()
    item = item.to_dict()
    return {"data":item}

    
# pdf crud
@app.post("/create_pdf/{id}" , tags=["PDF"])
def create_pdf(id:int, content:str, pdf_generator: PDFGenerator = Depends(get_pdf_generator)):
    pdf_generator.create_pdf(content)
    return {"message": "Pdf created successfully", "file":pdf_generator.filename}

@app.get("/get_pdf/{id}" , tags=["PDF"])
def get_pdf(id:int, pdf_generator: PDFGenerator = Depends(get_pdf_generator)):
    
    if not os.path.exists(pdf_generator.filename):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_generator.filename, media_type='application/pdf', filename=f"document_{id}.pdf")

@app.delete("/delete_pdf/{id}" , tags=["PDF"])
def delete_pdf(id:int, pdf_generator:PDFGenerator = Depends(get_pdf_generator)):
    
    if not os.path.exists(pdf_generator.filename):
        raise HTTPException(status_code=404, detail='file does not exist')
    pdf_generator.delete_pdf()
    return {"message":"pdf deleted successfully", "file": pdf_generator.filename}

@app.put("/update_pdf/{id}" , tags=["PDF"])
def update_pdf(id: int, content:str, pdf_generator: PDFGenerator = Depends(get_pdf_generator)):
    if not os.path.exists(pdf_generator.filename):
        raise HTTPException(status_code=404, detail="file is not found")
    pdf_generator.delete_pdf()
    pdf_generator.create_pdf(content)
    return {"message":"you have updated the pdf", "file": pdf_generator.filename}


