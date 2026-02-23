from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
import os

# ======================
# CONFIG
# ======================

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")  # FIX: load from env in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DATABASE_URL = "sqlite:///./database.db"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)

# ======================
# DATABASE MODELS
# ======================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    deadline = Column(DateTime)
    priority = Column(String)
    status = Column(String, default="Pending")
    category = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))


Base.metadata.create_all(bind=engine)

# ======================
# SCHEMAS
# ======================

class UserCreate(BaseModel):
    username: str
    password: str


class LoginSchema(BaseModel):
    username: str
    password: str


class TaskCreate(BaseModel):
    title: str
    deadline: datetime

    @field_validator('deadline')
    @classmethod
    def strip_timezone(cls, v):
        # Convert to naive UTC so it's consistent with datetime.utcnow()
        if v.tzinfo is not None:
            v = v.replace(tzinfo=None)
        return v


class TaskOut(BaseModel):
    id: int
    title: str
    deadline: datetime
    priority: str
    status: str
    category: str

    class Config:
        from_attributes = True  # FIX: added response schema for proper serialization


# ======================
# DATABASE DEPENDENCY
# ======================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================
# AUTH HELPERS
# ======================

def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def now_utc():
    return datetime.utcnow()  # naive UTC — consistent with SQLite and Pydantic on Python 3.11


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = now_utc() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},  # FIX: required by OAuth2 spec
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(
        User.username == username
    ).first()

    if user is None:
        raise credentials_exception

    return user

# ======================
# AI LOGIC
# ======================

def detect_category(title):
    title = title.lower()

    if any(word in title for word in ["study", "exam", "revision", "dsa"]):
        return "Study"
    if any(word in title for word in ["project", "code", "develop"]):
        return "Work"
    if any(word in title for word in ["gym", "exercise", "run"]):
        return "Health"
    return "General"


def calculate_priority(deadline, title):
    score = 0
    days_left = (deadline - now_utc()).days  # FIX: use now_utc() instead of utcnow()

    if days_left <= 1:
        score += 3
    elif days_left <= 3:
        score += 2
    else:
        score += 1

    urgent_words = ["urgent", "important", "asap", "interview"]
    if any(word in title.lower() for word in urgent_words):
        score += 2

    if score >= 4:
        return "High"
    elif score >= 3:
        return "Medium"
    return "Low"

# ======================
# ROUTES
# ======================

@app.get("/")
def home():
    return {"message": "AI Smart Task Manager Running"}


# REGISTER
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):

    existing = db.query(User).filter(
        User.username == user.username
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed = hash_password(user.password)

    new_user = User(
        username=user.username,
        password=hashed
    )

    db.add(new_user)
    db.commit()

    return {"message": "User registered successfully"}


# LOGIN (JSON BASED)
@app.post("/login")
def login(data: LoginSchema,
          db: Session = Depends(get_db)):

    user = db.query(User).filter(
        User.username == data.username
    ).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# CREATE TASK
@app.post("/tasks", response_model=dict)
def create_task(task: TaskCreate,
                user=Depends(get_current_user),
                db: Session = Depends(get_db)):

    category = detect_category(task.title)
    priority = calculate_priority(task.deadline, task.title)

    new_task = Task(
        title=task.title,
        deadline=task.deadline,
        priority=priority,
        category=category,
        status="Pending",
        user_id=user.id
    )

    db.add(new_task)
    db.commit()

    return {
        "message": "Task added",
        "priority": priority,
        "category": category
    }


# GET TASKS
@app.get("/tasks", response_model=list[TaskOut])  # FIX: added response_model for proper serialization
def get_tasks(user=Depends(get_current_user),
              db: Session = Depends(get_db)):

    tasks = db.query(Task).filter(
        Task.user_id == user.id
    ).all()

    return tasks


# COMPLETE TASK
@app.put("/tasks/{task_id}/complete")
def complete_task(task_id: int,
                  user=Depends(get_current_user),
                  db: Session = Depends(get_db)):

    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == user.id
    ).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = "Completed"
    db.commit()

    return {"message": "Task completed"}


# DELETE TASK  # FIX: added missing delete endpoint
@app.delete("/tasks/{task_id}")
def delete_task(task_id: int,
                user=Depends(get_current_user),
                db: Session = Depends(get_db)):

    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == user.id
    ).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()

    return {"message": "Task deleted"}


# DASHBOARD
@app.get("/dashboard")
def dashboard(user=Depends(get_current_user),
              db: Session = Depends(get_db)):

    tasks = db.query(Task).filter(
        Task.user_id == user.id
    ).all()

    total = len(tasks)
    completed = len([t for t in tasks if t.status == "Completed"])
    pending = total - completed
    overdue = len([
        t for t in tasks
        if t.deadline < now_utc() and t.status != "Completed"  # FIX: use now_utc()
    ])

    productivity = int((completed / total) * 100) if total > 0 else 100

    overload_risk = "Low"
    if pending > 5:
        overload_risk = "High"
    elif pending > 3:
        overload_risk = "Medium"

    return {
        "total_tasks": total,
        "completed": completed,
        "pending": pending,
        "overdue": overdue,
        "productivity_score": productivity,
        "overload_risk": overload_risk
    }