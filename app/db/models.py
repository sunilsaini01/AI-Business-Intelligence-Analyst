"""SQLAlchemy 2.0 models for both schemas on the one Postgres instance (Sec 3).

`analytics.*`  — seeded business data, SELECT-only to the `readonly_analyst` role.
                 The SQL Agent may only ever read these tables.
`app.*`        — session/report/eval state, read/write via the app role.
                 Never touched by LLM-generated SQL.

Role-level GRANTs enforcing this split live in the Alembic migration, not here —
this file only describes shape.
"""

from __future__ import annotations

import datetime
import enum
import uuid

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ======================================================================
# analytics schema — seeded, read-only to the LLM
# ======================================================================


class Region(Base):
    __tablename__ = "regions"
    __table_args__ = {"schema": "analytics"}

    region_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)

    customers: Mapped[list["Customer"]] = relationship(back_populates="region")
    orders: Mapped[list["Order"]] = relationship(back_populates="region")


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("customer_id"),
        {"schema": "analytics"},
    )

    customer_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    segment: Mapped[str] = mapped_column(String(50), index=True)  # SMB | Mid-Market | Enterprise
    region_id: Mapped[int] = mapped_column(ForeignKey("analytics.regions.region_id"), index=True)
    signup_date: Mapped[datetime.date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default="active")

    region: Mapped[Region] = relationship(back_populates="customers")
    orders: Mapped[list["Order"]] = relationship(back_populates="customer")
    activity: Mapped[list["CustomerActivity"]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = {"schema": "analytics"}

    product_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(100), index=True)
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2))
    cost: Mapped[float] = mapped_column(Numeric(10, 2))


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = {"schema": "analytics"}

    order_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("analytics.customers.customer_id"), index=True
    )
    region_id: Mapped[int] = mapped_column(ForeignKey("analytics.regions.region_id"), index=True)
    order_date: Mapped[datetime.date] = mapped_column(Date, index=True)
    channel: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="completed")

    customer: Mapped[Customer] = relationship(back_populates="orders")
    region: Mapped[Region] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order")
    payments: Mapped[list["Payment"]] = relationship(back_populates="order")


class OrderItem(Base):
    """revenue = SUM(quantity * unit_price * (1 - discount)) — computed in pandas, never by the LLM."""

    __tablename__ = "order_items"
    __table_args__ = {"schema": "analytics"}

    order_item_id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("analytics.orders.order_id"), index=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("analytics.products.product_id"), index=True
    )
    quantity: Mapped[int]
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2))
    discount: Mapped[float] = mapped_column(Numeric(4, 3), default=0)

    order: Mapped[Order] = relationship(back_populates="items")


class Payment(Base):
    """Separates 'ordered' from 'paid' — enables an AOV-vs-collected distinction."""

    __tablename__ = "payments"
    __table_args__ = {"schema": "analytics"}

    payment_id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("analytics.orders.order_id"), index=True)
    amount: Mapped[float] = mapped_column(Numeric(10, 2))
    method: Mapped[str] = mapped_column(String(50))
    paid_at: Mapped[datetime.datetime] = mapped_column(DateTime)

    order: Mapped[Order] = relationship(back_populates="payments")


class MarketingCampaign(Base):
    __tablename__ = "marketing_campaigns"
    __table_args__ = {"schema": "analytics"}

    campaign_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    region_id: Mapped[int] = mapped_column(ForeignKey("analytics.regions.region_id"), index=True)
    start_date: Mapped[datetime.date] = mapped_column(Date, index=True)
    end_date: Mapped[datetime.date] = mapped_column(Date)
    spend: Mapped[float] = mapped_column(Numeric(12, 2))
    channel: Mapped[str] = mapped_column(String(50))


class CustomerActivity(Base):
    """login / support-ticket / cart-abandon events — the churn model's feature source."""

    __tablename__ = "customer_activity"
    __table_args__ = {"schema": "analytics"}

    activity_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("analytics.customers.customer_id"), index=True
    )
    activity_date: Mapped[datetime.date] = mapped_column(Date, index=True)
    activity_type: Mapped[str] = mapped_column(String(50))  # login | support_ticket | cart_abandon

    customer: Mapped[Customer] = relationship(back_populates="activity")


# ======================================================================
# app schema — SQLAlchemy-managed, read/write, never touched by LLM SQL
# ======================================================================


class SessionStatus(str, enum.Enum):
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    DONE = "DONE"
    FAILED = "FAILED"


class ConfidenceLevel(str, enum.Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class User(Base):
    """Auth-ready, not wired to a login flow yet — a config flag away, not a redesign."""

    __tablename__ = "users"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="analyst")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class AnalysisSession(Base):
    """The row the status endpoint (Sec 7) polls."""

    __tablename__ = "analysis_sessions"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("app.users.id"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[SessionStatus] = mapped_column(
        String(20), default=SessionStatus.PENDING, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Phase 13, Objective D: structured execution observability — one
    # additive JSONB column (mirroring AnalysisReport.report_extras' proven
    # shape, migration 0003) rather than a new table. Populated once, at the
    # end of app/services/analysis_service.py::run_analysis (DONE or
    # FAILED) — see that function's `_build_execution_metadata` for the
    # exact keys. Never contains secrets: node names, counts, timestamps,
    # booleans, and token counts only.
    execution_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    steps: Mapped[list["AnalysisStep"]] = relationship(back_populates="session")
    report: Mapped["AnalysisReport | None"] = relationship(back_populates="session", uselist=False)
    charts: Mapped[list["Chart"]] = relationship(back_populates="session")


class AnalysisStep(Base):
    """Append-only trace: this *is* the observability layer (Sec 9) — no separate logging DB."""

    __tablename__ = "analysis_steps"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("app.analysis_sessions.id"), index=True
    )
    agent_name: Mapped[str] = mapped_column(String(100))
    step_order: Mapped[int]
    input_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    output_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20))
    duration_ms: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped[AnalysisSession] = relationship(back_populates="steps")


class AnalysisReport(Base):
    __tablename__ = "analysis_reports"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("app.analysis_sessions.id"), unique=True, index=True
    )
    executive_summary: Mapped[str] = mapped_column(Text)
    key_findings: Mapped[list] = mapped_column(JSONB, default=list)
    evidence: Mapped[list] = mapped_column(JSONB, default=list)
    recommendations: Mapped[list] = mapped_column(JSONB, default=list)
    confidence: Mapped[ConfidenceLevel] = mapped_column(String(10))
    limitations: Mapped[str] = mapped_column(Text, default="")
    # Phase 10 (Report Generator): the 5 presentation-only fields it adds
    # (verified_claims, analysis_explanation, visualizations,
    # technical_details, narrative) — one additive JSONB column rather than
    # 5 new ones, since they're always read/written together as a unit and
    # never queried individually. See app/graph/state.py::BusinessReport
    # and migration 0003_report_extras.
    report_extras: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped[AnalysisSession] = relationship(back_populates="report")


class Chart(Base):
    __tablename__ = "charts"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("app.analysis_sessions.id"), index=True
    )
    chart_type: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(300))
    storage_path: Mapped[str] = mapped_column(Text)
    spec_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped[AnalysisSession] = relationship(back_populates="charts")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    label: Mapped[str] = mapped_column(String(200))
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model_name: Mapped[str] = mapped_column(String(100))
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    aggregate_scores: Mapped[dict] = mapped_column(JSONB, default=dict)

    results: Mapped[list["EvaluationResult"]] = relationship(back_populates="run")


class EvaluationResult(Base):
    """One row per benchmark case per run — what makes a regression diffable, not anecdotal."""

    __tablename__ = "evaluation_results"
    __table_args__ = {"schema": "app"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("app.evaluation_runs.id"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(100))
    scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    judge_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(nullable=True)
    passed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped[EvaluationRun] = relationship(back_populates="results")
