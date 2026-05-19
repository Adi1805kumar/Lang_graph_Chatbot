import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Text,
    Integer, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./chatbot.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship(
        "Message", back_populates="session", cascade="all, delete"
    )
    documents = relationship(
        "Document", back_populates="session", cascade="all, delete"
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)   # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="documents")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Helper functions used directly by app.py ─────────────────────────────────

def create_session(title="New Chat") -> ChatSession:
    db = SessionLocal()
    session = ChatSession(id=str(uuid.uuid4()), title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    db.close()
    return session


def get_all_sessions() -> list[ChatSession]:
    db = SessionLocal()
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
    # detach safely
    result = [
        {"id": s.id, "title": s.title, "created_at": s.created_at,
         "message_count": len(s.messages)}
        for s in sessions
    ]
    db.close()
    return result


def delete_session(session_id: str):
    db = SessionLocal()
    s = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if s:
        db.delete(s)
        db.commit()
    db.close()


def get_messages(session_id: str) -> list[dict]:
    db = SessionLocal()
    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.id)
        .all()
    )
    result = [{"role": m.role, "content": m.content} for m in msgs]
    db.close()
    return result


def save_message(session_id: str, role: str, content: str):
    db = SessionLocal()
    msg = Message(session_id=session_id, role=role, content=content)
    db.add(msg)
    # Auto-title: set session title to first user message
    if role == "user":
        s = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if s and s.title == "New Chat":
            s.title = content[:45] + ("..." if len(content) > 45 else "")
    db.commit()
    db.close()


def get_documents(session_id: str) -> list[dict]:
    db = SessionLocal()
    docs = db.query(Document).filter(Document.session_id == session_id).all()
    result = [{"id": d.id, "filename": d.filename} for d in docs]
    db.close()
    return result


def save_document(session_id: str, filename: str):
    db = SessionLocal()
    doc = Document(session_id=session_id, filename=filename)
    db.add(doc)
    db.commit()
    db.close()