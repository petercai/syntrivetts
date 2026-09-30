from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True)
    title = Column(String(512), nullable=False)
    author = Column(String(512))
    publisher = Column(String(256))
    language = Column(String(4))
    publish_date = Column(String(64))
    original_filename = Column(String(256))
    cover = Column(String(1024))
    created_at = Column(DateTime, default=datetime.utcnow)

    extraction_mode = Column(String(32), nullable=True, default="none")

    jobs = relationship("Job", back_populates="book")
    images = relationship(
        "BookImage", back_populates="book", cascade="all, delete-orphan"
    )


class BookImage(Base):
    __tablename__ = "book_images"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    name = Column(String(256))
    path = Column(String(1024))
    image_type = Column(String(32), default="internal")

    book = relationship("Book", back_populates="images")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    process_dir = Column(String(1024), nullable=False)
    epub_path = Column(String(1024), nullable=False)
    cover_path = Column(String(1024))
    stage = Column(String(64), default="init")
    status = Column(String(32), default="pending")
    synthesis_mode = Column(String(32))
    chapter_audio_dir = Column(String(1024))
    sentence_audio_dir = Column(String(1024))
    audiobooks_dir = Column(String(1024))
    transcript_dir = Column(String(1024), nullable=True)
    current_step = Column(String(64))
    merge_mode_override = Column(String(32), nullable=True, default=None)
    chapter_number_reset_per_volume = Column(Boolean, nullable=True, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    archived_at = Column(DateTime, nullable=True)

    book = relationship("Book", back_populates="jobs")
    tts_config = relationship(
        "TtsConfig", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    output_config = relationship(
        "OutputConfig", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    stage_events = relationship(
        "StageEvent",
        back_populates="job",
        order_by="StageEvent.started_at",
        cascade="all, delete-orphan",
    )
    transcript_chapters = relationship(
        "TranscriptChapter", back_populates="job", cascade="all, delete-orphan"
    )
    threshold_blocks = relationship(
        "ThresholdBlockEvent", back_populates="job", cascade="all, delete-orphan"
    )
    workflow_step_events = relationship(
        "WorkflowStepEvent",
        back_populates="job",
        order_by="WorkflowStepEvent.created_at",
        cascade="all, delete-orphan",
    )
    cleaning_rule_configs = relationship(
        "JobCleaningRuleConfig",
        back_populates="job",
        cascade="all, delete-orphan",
    )
    sentence_review_events = relationship(
        "SentenceReviewEvent",
        back_populates="job",
        cascade="all, delete-orphan",
    )


class TtsConfig(Base):
    __tablename__ = "tts_configs"
    __table_args__ = (UniqueConstraint("job_id"),)

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    engine = Column(String(64), default="cosyvoice")
    model = Column(String(128), nullable=True, default=None)
    language = Column(String(4))
    device = Column(String(16))
    offline_mode = Column(Boolean, default=True)
    voice_dir = Column(String(1024), default="voices")
    fine_tuned_model = Column(String(256), default="internal")
    temperature = Column(Float, default=0.05)
    length_penalty = Column(Float, default=1.0)
    num_beams = Column(Integer, default=1)
    repetition_penalty = Column(Float, default=1.15)
    top_k = Column(Integer, default=40)
    top_p = Column(Float, default=0.85)
    speed = Column(Float, default=1.0)
    enable_text_splitting = Column(Boolean, default=False)
    text_temp = Column(Float, default=0.5)
    waveform_temp = Column(Float, default=0.5)
    cfg_value = Column(Float, default=2.0)
    inference_timesteps = Column(Integer, default=10)
    normalize = Column(Boolean, default=True)
    denoise = Column(Boolean, default=True)
    retry_badcase = Column(Boolean, default=True)
    retry_badcase_max_times = Column(Integer, default=3)
    retry_badcase_ratio_threshold = Column(Float, default=6.0)

    job = relationship("Job", back_populates="tts_config")
    tts_voices = relationship(
        "TtsVoice", back_populates="tts_config", cascade="all, delete-orphan"
    )


class OutputConfig(Base):
    __tablename__ = "output_configs"
    __table_args__ = (UniqueConstraint("job_id"),)

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    output_format = Column(String(16), default="m4a")
    output_split = Column(String(32), default="by-chapter")
    output_split_minutes = Column(Integer, default=30)

    job = relationship("Job", back_populates="output_config")


class StageEvent(Base):
    __tablename__ = "stage_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    stage_name = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    artifact_paths = Column(JSON)

    job = relationship("Job", back_populates="stage_events")


class TranscriptChapter(Base):
    __tablename__ = "transcript_chapters"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    chapter_id = Column(String(64), nullable=False)
    group_id = Column(String(64))
    title = Column(String(512))
    source_href = Column(String(1024))
    raw_html_path = Column(String(1024))
    cleaned_html_path = Column(String(1024))
    human_edited_html_path = Column(String(1024))
    para_id_map_path = Column(String(1024))
    transcript_path = Column(String(1024))
    transcript_lines = Column(Integer)
    volume = Column(String(512))
    volume_number = Column(String(16))
    sequence_number = Column(String(16))
    chapter_name = Column(String(512))
    chapter_number = Column(String(16))
    raw_chars = Column(Integer)
    cleaned_chars = Column(Integer)
    deletion_ratio = Column(Float)
    threshold_blocked = Column(Boolean, default=False)
    threshold_override = Column(Boolean, default=False)
    human_edited = Column(Boolean, default=False)
    synthesis_status = Column(String(32), default="pending")
    synthesis_batch_id = Column(Integer, ForeignKey("synthesis_batches.synthesis_batch_id"), nullable=True)
    chapter_audio_path = Column(String(1024), nullable=True)
    chapter_audio_seconds = Column(Float, nullable=True)

    job = relationship("Job", back_populates="transcript_chapters")
    synthesis_batch = relationship("SynthesisBatch", back_populates="chapters")


class SynthesisBatch(Base):
    __tablename__ = "synthesis_batches"

    synthesis_batch_id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    note = Column(String(512), nullable=True)
    synth_error = Column(String(512), nullable=True)
    failed_line_index = Column(Integer, nullable=True)
    synth_started_at = Column(DateTime, nullable=True)
    synth_finished_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)

    chapters = relationship("TranscriptChapter", back_populates="synthesis_batch")

    def __repr__(self) -> str:
        return f"<SynthesisBatch synthesis_batch_id={self.synthesis_batch_id} synth_error={self.synth_error!r}>"


class ThresholdBlockEvent(Base):
    __tablename__ = "threshold_block_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    chapter_id = Column(String(64))
    deletion_ratio = Column(Float, nullable=False)
    rule_distribution = Column(JSON)
    override_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="threshold_blocks")


class WorkflowStepEvent(Base):
    __tablename__ = "workflow_step_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    step_name = Column(String(64), nullable=False)
    action = Column(String(32), nullable=False)
    artifacts_summary = Column(JSON)
    user_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="workflow_step_events")


class CleaningRule(Base):
    __tablename__ = "cleaning_rules"

    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=False, unique=True)
    rule_class = Column(String(256), nullable=False)
    display_name = Column(String(256))
    description = Column(Text)
    sort_order = Column(Integer, nullable=False)
    enabled_by_default = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class JobCleaningRuleConfig(Base):
    __tablename__ = "job_cleaning_rule_configs"
    __table_args__ = (UniqueConstraint("job_id", "rule_name"),)

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    rule_name = Column(String(64), nullable=False)
    enabled = Column(Boolean, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    job = relationship("Job", back_populates="cleaning_rule_configs")


class ReferenceVoice(Base):
    __tablename__ = "reference_voices"
    __table_args__ = (UniqueConstraint("path"),)

    id = Column(Integer, primary_key=True)
    path = Column(String(1024), nullable=False)
    name = Column(String(256), nullable=False)
    gender = Column(String(16), nullable=False)
    language = Column(String(4), nullable=False)
    accent = Column(String(64), nullable=True)
    description = Column(String(512), nullable=True)
    sample_rate = Column(Integer, nullable=True)
    bit_depth = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tags = relationship(
        "ReferenceVoiceTag", back_populates="reference_voice", cascade="all, delete-orphan"
    )
    tts_voices = relationship("TtsVoice", back_populates="reference_voice")

    def __repr__(self) -> str:
        return f"<ReferenceVoice id={self.id} path={self.path!r} name={self.name!r}>"


class ReferenceVoiceTag(Base):
    __tablename__ = "reference_voice_tags"
    __table_args__ = (UniqueConstraint("reference_voice_id", "tag"),)

    id = Column(Integer, primary_key=True)
    reference_voice_id = Column(Integer, ForeignKey("reference_voices.id"), nullable=False)
    tag = Column(String(128), nullable=False)

    reference_voice = relationship("ReferenceVoice", back_populates="tags")

    def __repr__(self) -> str:
        return f"<ReferenceVoiceTag reference_voice_id={self.reference_voice_id} tag={self.tag!r}>"


class TtsVoice(Base):
    __tablename__ = "tts_voices"
    __table_args__ = (UniqueConstraint("tts_config_id", "voice_id"),)

    id = Column(Integer, primary_key=True)
    tts_config_id = Column(Integer, ForeignKey("tts_configs.id"), nullable=False)
    name = Column(String(256), nullable=False)
    voice_id = Column(Integer, nullable=False)
    reference_voice_id = Column(Integer, ForeignKey("reference_voices.id"), nullable=True)
    excluded = Column(Boolean, default=False)

    tts_config = relationship("TtsConfig", back_populates="tts_voices")
    reference_voice = relationship("ReferenceVoice", back_populates="tts_voices")

    def __repr__(self) -> str:
        return (
            f"<TtsVoice tts_config_id={self.tts_config_id} name={self.name!r}"
            f" voice_id={self.voice_id} excluded={self.excluded}>"
        )


class SentenceReviewEvent(Base):
    __tablename__ = "sentence_review_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    transcript_chapter_id = Column(Integer, ForeignKey("transcript_chapters.id"), nullable=False)
    sentence_index = Column(Integer, nullable=False)
    action = Column(String(32), nullable=False)
    issue_category = Column(String(32), nullable=True)
    description_tag_snapshot = Column(String(128), nullable=True)
    trash_path = Column(String(1024), nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    job = relationship("Job", back_populates="sentence_review_events")
    transcript_chapter = relationship("TranscriptChapter")

    def __repr__(self) -> str:
        return (
            f"<SentenceReviewEvent job_id={self.job_id}"
            f" transcript_chapter_id={self.transcript_chapter_id}"
            f" sentence_index={self.sentence_index} action={self.action!r}>"
        )


class JobLease(Base):
    __tablename__ = "job_leases"

    job_id = Column(Integer, primary_key=True)
    holder_id = Column(String(64), nullable=False)
    holder_kind = Column(String(32), nullable=False)
    operation = Column(String(128), nullable=True)
    pid = Column(Integer, nullable=False)
    hostname = Column(String(255), nullable=False)
    acquired_at = Column(Float, nullable=False)
    heartbeat_at = Column(Float, nullable=False)
    expires_at = Column(Float, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<JobLease job_id={self.job_id} holder={self.holder_kind}:{self.holder_id[:8]}"
            f" pid={self.pid}@{self.hostname} op={self.operation!r}>"
        )
