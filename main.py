from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from typing import Optional
import os

# ======================
# CONFIG
# ======================

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = "sqlite:///./database.db"

app = FastAPI(title="TaskFlow API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ======================
# DATABASE MODELS
# ======================

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True)
    username   = Column(String, unique=True, index=True)
    password   = Column(String)
    tasks      = relationship("Task", back_populates="owner", cascade="all, delete")
    shared_tasks = relationship("SharedTask", foreign_keys="SharedTask.shared_with_id", back_populates="shared_with_user")


class Task(Base):
    __tablename__ = "tasks"
    id          = Column(Integer, primary_key=True)
    title       = Column(String)
    notes       = Column(Text, default="")
    deadline    = Column(DateTime)
    priority    = Column(String)
    status      = Column(String, default="Pending")
    category    = Column(String)
    user_id     = Column(Integer, ForeignKey("users.id"))
    owner       = relationship("User", back_populates="tasks")
    subtasks    = relationship("Subtask", back_populates="task", cascade="all, delete")
    shared_with = relationship("SharedTask", back_populates="task", cascade="all, delete")


class Subtask(Base):
    __tablename__ = "subtasks"
    id        = Column(Integer, primary_key=True)
    title     = Column(String)
    done      = Column(Boolean, default=False)
    task_id   = Column(Integer, ForeignKey("tasks.id"))
    task      = relationship("Task", back_populates="subtasks")


class SharedTask(Base):
    __tablename__ = "shared_tasks"
    id              = Column(Integer, primary_key=True)
    task_id         = Column(Integer, ForeignKey("tasks.id"))
    shared_with_id  = Column(Integer, ForeignKey("users.id"))
    task            = relationship("Task", back_populates="shared_with")
    shared_with_user = relationship("User", foreign_keys=[shared_with_id], back_populates="shared_tasks")


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

class ChangePasswordSchema(BaseModel):
    old_password: str
    new_password: str

class TaskCreate(BaseModel):
    title: str
    deadline: datetime
    notes: Optional[str] = ""

    @field_validator('deadline')
    @classmethod
    def strip_timezone(cls, v):
        if v.tzinfo is not None:
            v = v.replace(tzinfo=None)
        return v

class TaskUpdate(BaseModel):
    title:    Optional[str] = None
    notes:    Optional[str] = None
    deadline: Optional[datetime] = None
    status:   Optional[str] = None

    @field_validator('deadline')
    @classmethod
    def strip_timezone(cls, v):
        if v and v.tzinfo is not None:
            v = v.replace(tzinfo=None)
        return v

class SubtaskCreate(BaseModel):
    title: str

class ShareTaskSchema(BaseModel):
    username: str  # username to share with

class SubtaskOut(BaseModel):
    id: int
    title: str
    done: bool
    class Config:
        from_attributes = True

class TaskOut(BaseModel):
    id:       int
    title:    str
    notes:    str
    deadline: datetime
    priority: str
    status:   str
    category: str
    subtasks: list[SubtaskOut] = []
    class Config:
        from_attributes = True

# ======================
# DB DEPENDENCY
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
    return datetime.utcnow()

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode.update({"exp": now_utc() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    exc = HTTPException(status_code=401, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise exc
    except JWTError:
        raise exc
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise exc
    return user

# ======================
# AI LOGIC
# ======================

def detect_category(title: str) -> str:
    t = title.lower()
    if any(w in t for w in ["study", "exam", "revision", "dsa", "learn", "course", "lecture"]):
        return "Study"
    if any(w in t for w in ["project", "code", "develop", "deploy", "debug", "meeting", "client"]):
        return "Work"
    if any(w in t for w in ["gym", "exercise", "run", "workout", "yoga", "diet", "health"]):
        return "Health"
    if any(w in t for w in ["buy", "shop", "order", "grocery", "market"]):
        return "Shopping"
    if any(w in t for w in ["call", "birthday", "family", "friend", "party", "event"]):
        return "Personal"
    return "General"

def calculate_priority(deadline: datetime, title: str) -> str:
    score = 0
    days_left = (deadline - now_utc()).days
    if days_left < 0:
        score += 4
    elif days_left <= 1:
        score += 3
    elif days_left <= 3:
        score += 2
    else:
        score += 1
    if any(w in title.lower() for w in ["urgent", "important", "asap", "interview", "critical", "deadline"]):
        score += 2
    return "High" if score >= 4 else "Medium" if score >= 3 else "Low"

def smart_suggestions(tasks: list) -> list[str]:
    suggestions = []
    pending = [t for t in tasks if t.status == "Pending"]
    overdue = [t for t in tasks if t.deadline < now_utc() and t.status != "Completed"]
    high    = [t for t in pending if t.priority == "High"]

    if overdue:
        suggestions.append(f"⚠ You have {len(overdue)} overdue task(s). Address them first!")
    if len(high) >= 3:
        suggestions.append(f"🔥 {len(high)} high-priority tasks pending — consider delegating some.")
    if len(pending) > 8:
        suggestions.append("📦 Your task list is overloaded. Try completing small tasks first.")
    if not overdue and len(pending) <= 3:
        suggestions.append("✅ You're in great shape! Keep the momentum going.")

    work  = sum(1 for t in pending if t.category == "Work")
    study = sum(1 for t in pending if t.category == "Study")
    if work > 5:
        suggestions.append("💼 Heavy work load detected. Schedule focused deep-work blocks.")
    if study > 3:
        suggestions.append("📚 Multiple study tasks pending. Try the Pomodoro technique!")
    return suggestions

# ======================
# ROUTES
# ======================

@app.get("/")
def home():
    return {"message": "TaskFlow API v2.0 Running"}

# --- AUTH ---

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    db.add(User(username=user.username, password=hash_password(user.password)))
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer"}

@app.put("/change-password")
def change_password(data: ChangePasswordSchema, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(data.old_password, user.password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    user.password = hash_password(data.new_password)
    db.commit()
    return {"message": "Password changed successfully"}

# --- TASKS ---

@app.post("/tasks", response_model=dict)
def create_task(task: TaskCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    category = detect_category(task.title)
    priority = calculate_priority(task.deadline, task.title)
    db.add(Task(
        title=task.title, notes=task.notes or "",
        deadline=task.deadline, priority=priority,
        category=category, status="Pending", user_id=user.id
    ))
    db.commit()
    return {"message": "Task added", "priority": priority, "category": category}

@app.get("/tasks", response_model=list[TaskOut])
def get_tasks(user=Depends(get_current_user), db: Session = Depends(get_db)):
    own    = db.query(Task).filter(Task.user_id == user.id).all()
    shared = [st.task for st in db.query(SharedTask).filter(SharedTask.shared_with_id == user.id).all()]
    return own + shared

@app.put("/tasks/{task_id}")
def update_task(task_id: int, data: TaskUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if data.title    is not None: task.title    = data.title
    if data.notes    is not None: task.notes    = data.notes
    if data.deadline is not None: task.deadline = data.deadline
    if data.status   is not None: task.status   = data.status
    # recalculate priority if title or deadline changed
    if data.title or data.deadline:
        task.priority = calculate_priority(task.deadline, task.title)
        task.category = detect_category(task.title)
    db.commit()
    return {"message": "Task updated", "priority": task.priority, "category": task.category}

@app.put("/tasks/{task_id}/complete")
def complete_task(task_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = "Completed"
    db.commit()
    return {"message": "Task completed"}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"message": "Task deleted"}

# --- SUBTASKS ---

@app.post("/tasks/{task_id}/subtasks")
def add_subtask(task_id: int, data: SubtaskCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    sub = Subtask(title=data.title, task_id=task_id)
    db.add(sub)
    db.commit()
    return {"message": "Subtask added", "id": sub.id}

@app.put("/subtasks/{subtask_id}/toggle")
def toggle_subtask(subtask_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    sub = db.query(Subtask).join(Task).filter(Subtask.id == subtask_id, Task.user_id == user.id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Subtask not found")
    sub.done = not sub.done
    db.commit()
    return {"message": "Toggled", "done": sub.done}

@app.delete("/subtasks/{subtask_id}")
def delete_subtask(subtask_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    sub = db.query(Subtask).join(Task).filter(Subtask.id == subtask_id, Task.user_id == user.id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Subtask not found")
    db.delete(sub)
    db.commit()
    return {"message": "Subtask deleted"}

# --- SHARING ---

@app.post("/tasks/{task_id}/share")
def share_task(task_id: int, data: ShareTaskSchema, user=Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    target = db.query(User).filter(User.username == data.username).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == user.id:
        raise HTTPException(status_code=400, detail="Cannot share with yourself")
    exists = db.query(SharedTask).filter(SharedTask.task_id == task_id, SharedTask.shared_with_id == target.id).first()
    if exists:
        raise HTTPException(status_code=400, detail="Already shared with this user")
    db.add(SharedTask(task_id=task_id, shared_with_id=target.id))
    db.commit()
    return {"message": f"Task shared with {data.username}"}

# --- DASHBOARD & STATS ---

@app.get("/dashboard")
def dashboard(user=Depends(get_current_user), db: Session = Depends(get_db)):
    tasks   = db.query(Task).filter(Task.user_id == user.id).all()
    total   = len(tasks)
    completed = len([t for t in tasks if t.status == "Completed"])
    pending   = total - completed
    overdue   = len([t for t in tasks if t.deadline < now_utc() and t.status != "Completed"])
    productivity = int((completed / total) * 100) if total > 0 else 100
    overload_risk = "High" if pending > 5 else "Medium" if pending > 3 else "Low"

    # Category breakdown
    cat_counts = {}
    for t in tasks:
        cat_counts[t.category] = cat_counts.get(t.category, 0) + 1

    # Weekly completion (last 7 days)
    week_data = []
    for i in range(6, -1, -1):
        day = now_utc() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0)
        day_end   = day.replace(hour=23, minute=59, second=59)
        count = len([t for t in tasks if t.status == "Completed" and day_start <= t.deadline <= day_end])
        week_data.append({"day": day.strftime("%a"), "count": count})

    suggestions = smart_suggestions(tasks)

    return {
        "total_tasks": total,
        "completed": completed,
        "pending": pending,
        "overdue": overdue,
        "productivity_score": productivity,
        "overload_risk": overload_risk,
        "category_breakdown": cat_counts,
        "weekly_data": week_data,
        "suggestions": suggestions,
    }

@app.get("/stats")
def stats(user=Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.user_id == user.id).all()
    priority_counts = {"High": 0, "Medium": 0, "Low": 0}
    for t in tasks:
        priority_counts[t.priority] = priority_counts.get(t.priority, 0) + 1
    upcoming = sorted(
        [t for t in tasks if t.status == "Pending" and t.deadline >= now_utc()],
        key=lambda t: t.deadline
    )[:5]
    return {
        "priority_breakdown": priority_counts,
        "upcoming_deadlines": [
            {"id": t.id, "title": t.title, "deadline": t.deadline.isoformat(), "priority": t.priority}
            for t in upcoming
        ]
    }

# --- REMINDER CHECK ---

@app.get("/reminders")
def get_reminders(user=Depends(get_current_user), db: Session = Depends(get_db)):
    soon = now_utc() + timedelta(hours=24)
    tasks = db.query(Task).filter(
        Task.user_id == user.id,
        Task.status == "Pending",
        Task.deadline <= soon,
        Task.deadline >= now_utc()
    ).all()
    overdue = db.query(Task).filter(
        Task.user_id == user.id,
        Task.status == "Pending",
        Task.deadline < now_utc()
    ).all()
    return {
        "due_soon": [{"id": t.id, "title": t.title, "deadline": t.deadline.isoformat()} for t in tasks],
        "overdue":  [{"id": t.id, "title": t.title, "deadline": t.deadline.isoformat()} for t in overdue],
    }


#deployment
import os

PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)