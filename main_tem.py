
import uvicorn
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy import create_engine, String, Column, Integer

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

app = FastAPI()

# 允许所有源的跨域请求
# app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # 或者["https://example.com"]等具体域名
#    allow_credentials=True,
#    allow_methods=["*"],  # 允许所有HTTP方法
#    allow_headers=["*"],  # 允许所有请求头
# )

# 创建对象的基类:
Base = declarative_base()

DBConfig = {
    "user": "root",
    "password": "ROOT",
    "host": "127.0.0.1",
    "port": 3306,
    "dbname": "ros2"
}


def get_db_session():
    connect_str = 'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}'.format(**DBConfig)
    print(connect_str)
    # 初始化数据库连接:
    engine = create_engine(connect_str, echo=True)
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    return DBSession()

class ReportModel(Base):
    # 表名字:
    __tablename__ = 'report'

    # 表的结构:
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    age = Column(String(100))

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    

class UserModel(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    email = Column(String)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
class A_ItemModel(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    price = Column(Integer)

    def to_dict(self):
        return { c.name: getattr(self, c.name) for c in self.__table__.columns}

class ReportInsertModel(BaseModel):
    name: str
    age: int

class UserInsertModel(BaseModel):
    name: str
    email: str

class ItemInsertModel(BaseModel):
    name: str
    price: int

@app.post("/insert_one")
def insert_one(report_record:ReportInsertModel, session: Session = Depends(get_db_session)):
    report = ReportModel(**dict(report_record))
    session.add(report)
    session.flush()
    data = report.to_dict()
    session.commit()
    session.close()
    return {"data": data}


@app.get("/fetch_one/{id}")
def fetch_one(id:int ,session: Session = Depends(get_db_session)):
    report = session.query(ReportModel).filter_by(id=id).first()
    session.close()
    return report

@app.delete("/delete_one/{id}")
def delete_one(id: int, session: Session = Depends(get_db_session)):
    report = session.query(ReportModel).filter_by(id=id).first()
    if report is None:
        session.close()
        raise HTTPException(status_code=404, detail="Item not found")
    session.delete(report)
    session.commit()
    session.close()
    return {"message": "Item deleted successfully"}

@app.get("/user/{id}")
def get_user(id:int, session: Session = Depends (get_db_session)):
    user = session.query(UserModel).filter_by(id = id).first()
    return user

@app.post("/add_user")
def add_user(user: UserInsertModel, session: Session = Depends(get_db_session)):
    user = UserModel(**dict(user))
    session.add(user)
    # session.flush()
    session.commit()
    data = user.to_dict()
    session.close()
    return {"data": data}

@app.delete("/delete/{id}")
def delete_user(id:int, session: Session = Depends(get_db_session)):
    user = session.query(UserModel).filter_by(id=id).first()
    session.delete(user)
    session.commit()
    session.close()

@app.put("/update/{id}")
def update_user(id: int, session: Session = Depends(get_db_session)):
    user = session.query(UserModel).filter_by(id=id).first()
    user.name = user.name + "123"
    session.commit()
    data = user.to_dict()
    session.close()
    return {"data": data}

# @app.get("/hello")
# def say_hello():
#     pass

@app.get("/get_item/{id}")
def get_item(id:int, session: Session = Depends(get_db_session)):
    item = session.query(A_ItemModel).filter_by(id=id).first()
    return item

@app.post("/add_item")
def add_item( item: ItemInsertModel,session: Session = Depends(get_db_session)):
    item = A_ItemModel(**dict(item))
    session.add(item)
    #  session.flush()
    session.commit()
    data = item.to_dict()
    session.close()
    
    return {"data": data}
   
@app.delete("/delete_item/{id}")
def delete_item(id:int, session: Session = Depends(get_db_session)):

    item = session.query(A_ItemModel).filter_by(id=id).first()
    session.delete(item)
    session.commit()
    session.close()
    return {"data": item}

@app.put("/update_item/{id}")
def update_item(id:int, session: Session = Depends(get_db_session)):
    item = session.query(A_ItemModel).filter_by(id=id).first()
    item.price = item.price*10
    session.commit()
    item = item.to_dict()
    session.close()
    return {"data":item}

if __name__ == '__main__':
    uvicorn.run("main:app")
