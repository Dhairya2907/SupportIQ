import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

log = logging.getLogger(__name__)

if settings.is_sqlite:
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
else:
    # Hosted Postgres (Neon/Supabase) closes idle connections; pre-ping avoids stale-connection errors.
    engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns() -> None:
    """Minimal forward-only migration: add any model column that the live table lacks.

    Enough for SQLite during development (ADD COLUMN only). Switch to Alembic when the
    schema needs renames, drops, or data migrations.
    """
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(dialect=engine.dialect)}"
            with engine.begin() as conn:
                conn.execute(text(ddl))
            log.info("Schema migrated: %s", ddl)


def init_db() -> None:
    from app import models  # noqa: F401  (register tables)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
