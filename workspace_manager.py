"""
workspace_manager.py
--------------------
Manages the per-job workspace structure for LectureScribe.

Every run creates a unique job folder under temp/:
    temp/<SourceName>__<YYYYMMDD_HHMMSS>/

Structure:
    metadata/   job.json, config.json, state.json, history.json
    audio/      extracted .wav files per lecture
    transcripts/ per-lecture .txt files + merged.txt
    prompts/    per-chunk prompt files (for debugging)
    ai/         per-chunk Nemotron JSON + merged.json
    diagrams/   rendered .png files
    pdf/        final PDF output
    logs/       pipeline.log
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parent / "temp"
HISTORY_DIR    = Path(__file__).parent / "temp" / "history"


# ── Job class ──────────────────────────────────────────────────────────────────
class Job:
    """
    Represents a single LectureScribe processing job.
    Holds all paths and metadata for one run.
    """

    def __init__(self, job_dir: Path):
        self.dir        = job_dir
        self.meta_dir   = job_dir / "metadata"
        self.audio_dir  = job_dir / "audio"
        self.trans_dir  = job_dir / "transcripts"
        self.prompt_dir = job_dir / "prompts"
        self.ai_dir     = job_dir / "ai"
        self.diagram_dir= job_dir / "diagrams"
        self.pdf_dir    = job_dir / "pdf"
        self.log_dir    = job_dir / "logs"

        # Metadata files
        self.job_json     = self.meta_dir / "job.json"
        self.config_json  = self.meta_dir / "config.json"
        self.state_json   = self.meta_dir / "state.json"
        self.history_json = self.meta_dir / "history.json"

        # Key output files
        self.merged_transcript = self.trans_dir / "merged.txt"
        self.merged_ai         = self.ai_dir    / "merged.json"
        self.log_file          = self.log_dir   / "pipeline.log"

    # ── Directory helpers ──────────────────────────────────────────────────────
    def create_dirs(self):
        """Create all subdirectories for this job."""
        for d in [
            self.meta_dir, self.audio_dir, self.trans_dir,
            self.prompt_dir, self.ai_dir, self.diagram_dir,
            self.pdf_dir, self.log_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Metadata helpers ───────────────────────────────────────────────────────
    def write_job(self, source: str, source_type: str, title: str):
        """Write job.json with basic job metadata."""
        data = {
            "title":       title,
            "source":      source,
            "source_type": source_type,   # "folder" | "youtube" | "url"
            "created_at":  datetime.now().isoformat(),
            "job_dir":     str(self.dir)
        }
        self._write_json(self.job_json, data)

    def write_config(self, config: dict):
        """Write config.json with pipeline settings."""
        self._write_json(self.config_json, config)

    def read_config(self) -> dict:
        """Read config.json."""
        return self._read_json(self.config_json)

    # ── State helpers ──────────────────────────────────────────────────────────
    def write_state(self, **kwargs):
        """
        Update state.json with current pipeline progress.
        Merges with existing state so callers only need to pass changed fields.
        """
        current = self._read_json(self.state_json) if self.state_json.exists() else {}
        current.update(kwargs)
        current["updated_at"] = datetime.now().isoformat()
        self._write_json(self.state_json, current)

    def read_state(self) -> dict:
        """Read current state.json."""
        if self.state_json.exists():
            return self._read_json(self.state_json)
        return {}

    def mark_step(self, step: str):
        """Convenience: update current pipeline step in state.json."""
        self.write_state(current_step=step)

    def mark_done(self):
        """Mark job as completed in state.json."""
        self.write_state(status="done", current_step="complete")

    def mark_failed(self, reason: str):
        """Mark job as failed in state.json."""
        self.write_state(status="failed", error=reason)

    # ── Per-file path helpers ──────────────────────────────────────────────────
    def audio_path(self, name: str) -> Path:
        """Path for an audio file: audio/<name>.wav"""
        return self.audio_dir / f"{name}.wav"

    def transcript_path(self, name: str) -> Path:
        """Path for a transcript file: transcripts/<name>.txt"""
        return self.trans_dir / f"{name}.txt"

    def prompt_path(self, chunk_index: int) -> Path:
        """Path for a prompt debug file: prompts/chunk<N>_prompt.txt"""
        return self.prompt_dir / f"chunk{chunk_index:03d}_prompt.txt"

    def ai_chunk_path(self, chunk_index: int) -> Path:
        """Path for a Nemotron chunk result: ai/chunk<N>.json"""
        return self.ai_dir / f"chunk{chunk_index:03d}.json"

    def diagram_path(self, block_index: int, label: str = "") -> Path:
        """Path for a rendered diagram PNG: diagrams/diagram_<N>_<label>.png"""
        safe = _sanitize(label)[:30]
        return self.diagram_dir / f"diagram_{block_index:03d}_{safe}.png"

    def pdf_path(self, title: str) -> Path:
        """Path for the final PDF: pdf/<title>.pdf"""
        safe = _sanitize(title)[:60]
        return self.pdf_dir / f"{safe}.pdf"

    # ── Logger ─────────────────────────────────────────────────────────────────
    def get_logger(self) -> logging.Logger:
        """Return a logger that writes to logs/pipeline.log."""
        logger = logging.getLogger(str(self.dir))
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fh)
        return logger

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _write_json(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _read_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ── Factory functions ──────────────────────────────────────────────────────────
def create_job(source: str, source_type: str, title: str, config: dict) -> Job:
    """
    Create a new job workspace.
    Returns a Job object with all dirs created and metadata written.
    """
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = _sanitize(title)[:50]
    job_folder = WORKSPACE_ROOT / f"{safe_title}__{timestamp}"

    job = Job(job_folder)
    job.create_dirs()
    job.write_job(source=source, source_type=source_type, title=title)
    job.write_config(config)
    job.write_state(status="created", current_step="init")

    return job


def list_jobs() -> list[dict]:
    """
    Scan temp/ and return a list of all jobs with their metadata.
    Sorted by creation time, newest first.
    """
    if not WORKSPACE_ROOT.exists():
        return []

    jobs = []
    for job_dir in WORKSPACE_ROOT.iterdir():
        if not job_dir.is_dir() or job_dir.name == "history":
            continue
        job = Job(job_dir)
        if not job.job_json.exists():
            continue
        try:
            meta  = job._read_json(job.job_json)
            state = job.read_state()
            jobs.append({
                "dir":        str(job_dir),
                "title":      meta.get("title", job_dir.name),
                "source":     meta.get("source", ""),
                "created_at": meta.get("created_at", ""),
                "status":     state.get("status", "unknown"),
                "step":       state.get("current_step", ""),
            })
        except Exception:
            continue

    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return jobs


def load_job(job_dir: str) -> Job:
    """Load an existing job from its directory path."""
    return Job(Path(job_dir))


# ── Utilities ──────────────────────────────────────────────────────────────────
def _sanitize(text: str) -> str:
    """Remove special characters for safe folder/file names."""
    return "".join(
        c if c.isalnum() or c in "._- " else ""
        for c in text
    ).replace(" ", "_")