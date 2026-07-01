# LectureScribe 🎓

> Transform lecture videos into structured, exam-ready PDF notes using AI.

**LectureScribe** is a local CLI tool that takes a lecture video (or folder of videos) and produces clean, color-coded PDF notes — complete with diagrams, key concepts, and critical callouts — powered by NVIDIA Nemotron-Ultra and OpenAI Whisper.

---

## Demo

**Input:** A folder of 15 Udemy Python lecture videos (with SRT subtitles)
**Output:** A 10-page structured PDF with color-coded notes, flowcharts, and critical callouts

| Cover Page | Notes with Diagrams |
|---|---|
| Title, subject, topics covered, read time | Headings · Body · Important · Critical boxes · Flowcharts |

---

## Features

- 🎵 **Audio extraction** from local MP4/MKV/AVI or any YouTube/direct URL via `yt-dlp`
- ⚡ **Instant transcription** from `.srt` subtitle files (no GPU needed)
- 🎙 **GPU transcription** via OpenAI Whisper `large-v3` on CUDA as fallback
- 🧠 **AI note generation** using NVIDIA Nemotron-Ultra 550B with structured JSON output
- 📊 **Auto diagram rendering** — Mermaid.js flowcharts embedded in the PDF
- 📄 **Styled PDF output** — color-coded by block type (heading/body/important/critical/note)
- 🖥 **Interactive TUI** — arrow-key navigation, folder browser, file preview table
- 📁 **Folder mode** — process an entire course day (multiple videos) into one PDF

---

## How It Works

```
Video / Folder / URL
        │
        ▼
  [1] Audio Extraction (ffmpeg / yt-dlp)
        │
        ▼
  [2] Transcription (SRT parser or Whisper large-v3 on GPU)
        │
        ▼
  [3] AI Analysis (NVIDIA Nemotron-Ultra → structured JSON)
        │
        ▼
  [4] Diagram Rendering (Mermaid CLI → PNG)
        │
        ▼
  [5] PDF Generation (ReportLab → styled A4 PDF)
```

---

## Installation

### Prerequisites

- Python 3.12 (via Miniconda recommended)
- NVIDIA GPU with CUDA (for Whisper; not needed if using SRT subtitles)
- [ffmpeg](https://ffmpeg.org/download.html) added to PATH
- [Node.js](https://nodejs.org/) + Mermaid CLI (`npm install -g @mermaid-js/mermaid-cli`)
- NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)

### Setup

```powershell
# Clone the repo
git clone https://github.com/vansh-kumar-007/lecturescribe.git
cd lecturescribe

# Create conda environment
conda create -n lecturescribe python=3.12 -y
conda activate lecturescribe

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install openai-whisper yt-dlp python-dotenv openai rich ffmpeg-python reportlab questionary pillow

# Add your NVIDIA API key
echo NVIDIA_API_KEY=your_key_here > .env
```

---

## Usage

```powershell
python main.py
```

The TUI will guide you through:

1. **Source type** — Local folder / YouTube URL / Other URL
2. **Folder navigation** — Browse to your course day folder
3. **Transcription mode** — SRT subtitles (instant) or Whisper GPU (accurate)
4. **File preview** — See all files that will be processed before confirming

The PDF opens automatically when done.

---

## Output Structure

Notes are color-coded by type:

| Type | Color | Description |
|---|---|---|
| `heading` | Dark blue, underlined | Major topic |
| `subheading` | Dark blue | Subtopic |
| `body` | Blue | Explanatory content |
| `important` | Red ★ | Key facts, formulas, definitions |
| `critical` | Purple box | Must-know concept (max 2/lecture) |
| `note` | Green ✎ | Mnemonics, examples, asides |
| `diagram` | — | Mermaid.js flowchart rendered as PNG |

---

## Project Structure

```
lecturescribe/
├── main.py              # Entry point + pipeline orchestration
├── tui.py               # Interactive terminal UI (questionary + rich)
├── utils.py             # Audio extraction, SRT parsing, chunking
├── transcriber.py       # Whisper GPU transcription
├── nemotron.py          # NVIDIA Nemotron API + JSON parsing + retry logic
├── diagram_renderer.py  # Mermaid CLI diagram rendering
├── pdf_renderer.py      # ReportLab PDF generation
├── prompts.py           # Nemotron system + user prompts
└── outputs/             # Generated PDFs, transcripts, diagrams
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Transcription | OpenAI Whisper `large-v3` (CUDA) |
| SRT parsing | Custom parser (no deps) |
| AI notes | NVIDIA Nemotron-Ultra 550B via NVIDIA NIM API |
| Diagrams | Mermaid.js CLI (`mmdc`) |
| PDF | ReportLab |
| TUI | `rich` + `questionary` |
| Audio | `ffmpeg` + `yt-dlp` |

---

## Limitations

- Nemotron API has a rate limit of ~32 requests/session; LectureScribe handles this with automatic retry + backoff
- Whisper `large-v3` transcription takes ~0.5x audio duration on an RTX 3050 — use SRT mode when subtitles are available
- Diagram rendering requires Node.js and Mermaid CLI installed globally

---

## License

MIT
