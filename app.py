import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(
    page_title="贵州话视频转文字 / 字幕工具",
    page_icon="🎬",
    layout="centered",
)


# =========================
# 基础工具
# =========================

def check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def run_command(command):
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def extract_audio(video_path: str, audio_path: str):
    """
    从视频中提取 16kHz 单声道 WAV 音频。
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    run_command(command)


def split_audio(audio_path: str, chunks_dir: str, chunk_seconds: int):
    """
    把长音频切成多个固定时长片段。
    输出文件格式：chunk_000.wav, chunk_001.wav ...
    """
    chunks_path = Path(chunks_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)

    output_pattern = str(chunks_path / "chunk_%03d.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        output_pattern,
    ]

    run_command(command)

    return sorted(chunks_path.glob("chunk_*.wav"))


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


@st.cache_resource
def load_model(model_size: str):
    return WhisperModel(
        model_size,
        device="auto",
        compute_type="auto",
    )


def transcribe_chunk(
    audio_path: str,
    model_size: str,
    language,
    initial_prompt: str | None,
):
    model = load_model(model_size)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
    )

    results = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })

    return results, info


def make_txt(segments) -> str:
    return "\n".join(item["text"] for item in segments)


def make_txt_with_time(segments) -> str:
    lines = []
    for item in segments:
        start = format_timestamp(item["start"])
        end = format_timestamp(item["end"])
        lines.append(f"[{start} --> {end}] {item['text']}")
    return "\n".join(lines)


def make_srt(segments) -> str:
    lines = []

    for index, item in enumerate(segments, start=1):
        start = format_timestamp(item["start"])
        end = format_timestamp(item["end"])

        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(item["text"])
        lines.append("")

    return "\n".join(lines)


def save_uploaded_file(uploaded_file, save_path: Path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def get_file_size_mb(uploaded_file) -> float:
    return uploaded_file.size / 1024 / 1024


# =========================
# 页面
# =========================

st.title("🎬 贵州话视频转文字 / 字幕工具")

st.write(
    "适合处理较长视频。系统会先提取音频，再把音频切成多个片段，逐段识别后合并结果。"
)

if not check_ffmpeg_available():
    st.error(
        "当前环境未检测到 FFmpeg。请确认 Streamlit Cloud 仓库根目录包含 `packages.txt`，"
        "并且里面写了 `ffmpeg`。"
    )
    st.stop()


# =========================
# 侧边栏参数
# =========================

st.sidebar.header("识别参数")

model_size = st.sidebar.selectbox(
    "Whisper 模型",
    ["small", "medium", "large-v3"],
    index=1,
    help="贵州话建议先用 medium。large-v3 更准但更慢，云端可能资源不足。",
)

language_option = st.sidebar.selectbox(
    "语言设置",
    ["中文", "自动识别"],
    index=0,
)

language_map = {
    "中文": "zh",
    "自动识别": None,
}

language = language_map[language_option]

chunk_minutes = st.sidebar.selectbox(
    "切片长度",
    [3, 5, 10],
    index=1,
    help="1 小时视频建议 5 分钟一段。机器配置较弱时选 3 分钟。",
)

initial_prompt = st.sidebar.text_area(
    "方言识别提示词",
    value=(
        "这是贵州话、西南官话、贵州方言内容。"
        "请尽量识别为中文，保留原意。"
        "常见语气词和口语可以按普通中文记录。"
    ),
    height=120,
)

st.sidebar.markdown("---")
st.sidebar.write("建议：")
st.sidebar.write("- 1 小时视频：切片 5 分钟")
st.sidebar.write("- 普通部署：先用 medium")
st.sidebar.write("- 口音重：再试 large-v3")
st.sidebar.write("- 不建议用 tiny/base 处理贵州话")


# =========================
# 上传与处理
# =========================

uploaded_file = st.file_uploader(
    "上传视频文件",
    type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
)

if uploaded_file is None:
    st.warning("请先上传一个视频文件。")
    st.stop()


file_size_mb = get_file_size_mb(uploaded_file)
file_stem = Path(uploaded_file.name).stem

st.write(f"文件名：`{uploaded_file.name}`")
st.write(f"文件大小：`{file_size_mb:.2f} MB`")

with st.expander("预览视频", expanded=False):
    st.video(uploaded_file)

start_button = st.button("开始转文字", type="primary")

if start_button:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        video_path = temp_dir / uploaded_file.name
        full_audio_path = temp_dir / "full_audio.wav"
        chunks_dir = temp_dir / "chunks"

        try:
            save_uploaded_file(uploaded_file, video_path)

            progress = st.progress(0)
            status = st.empty()

            status.write("正在从视频中提取音频...")
            progress.progress(5)

            extract_audio(str(video_path), str(full_audio_path))

            status.write("正在切分长音频...")
            progress.progress(12)

            chunk_seconds = chunk_minutes * 60
            chunk_paths = split_audio(
                str(full_audio_path),
                str(chunks_dir),
                chunk_seconds,
            )

            if not chunk_paths:
                st.error("音频切片失败，没有生成任何音频片段。")
                st.stop()

            all_segments = []
            detected_language = None
            detected_probability = None

            total_chunks = len(chunk_paths)

            for index, chunk_path in enumerate(chunk_paths):
                status.write(
                    f"正在识别第 {index + 1} / {total_chunks} 段..."
                )

                chunk_segments, info = transcribe_chunk(
                    str(chunk_path),
                    model_size,
                    language,
                    initial_prompt,
                )

                offset = index * chunk_seconds

                for item in chunk_segments:
                    all_segments.append({
                        "start": item["start"] + offset,
                        "end": item["end"] + offset,
                        "text": item["text"],
                    })

                if detected_language is None:
                    detected_language = info.language
                    detected_probability = info.language_probability

                current_progress = 12 + int((index + 1) / total_chunks * 78)
                progress.progress(min(current_progress, 90))

            status.write("正在生成结果文件...")
            progress.progress(95)

            txt_content = make_txt(all_segments)
            txt_with_time_content = make_txt_with_time(all_segments)
            srt_content = make_srt(all_segments)

            progress.progress(100)
            status.write("处理完成！")

            st.success("转写完成")

            st.write(f"识别语言：`{detected_language}`")
            if detected_probability is not None:
                st.write(f"语言置信度：`{detected_probability:.2f}`")

            st.write(f"音频切片数：`{total_chunks}`")
            st.write(f"字幕段落数：`{len(all_segments)}`")

            st.subheader("转写预览")
            st.text_area(
                "带时间戳文本",
                txt_with_time_content,
                height=350,
            )

            st.subheader("下载结果")

            st.download_button(
                label="下载 TXT 纯文本",
                data=txt_content,
                file_name=f"{file_stem}.txt",
                mime="text/plain",
            )

            st.download_button(
                label="下载 TXT 带时间戳",
                data=txt_with_time_content,
                file_name=f"{file_stem}_with_time.txt",
                mime="text/plain",
            )

            st.download_button(
                label="下载 SRT 字幕",
                data=srt_content,
                file_name=f"{file_stem}.srt",
                mime="text/plain",
            )

        except subprocess.CalledProcessError as e:
            error_message = (
                e.stderr.decode("utf-8", errors="ignore")
                if e.stderr
                else str(e)
            )

            st.error("FFmpeg 处理失败。")
            st.code(error_message)

        except RuntimeError as e:
            st.error("AI 模型加载或识别失败。")
            st.code(str(e))

        except Exception as e:
            st.error("处理失败。")
            st.code(str(e))
