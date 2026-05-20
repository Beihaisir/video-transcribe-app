import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="视频转文字 / 字幕工具",
    page_icon="🎬",
    layout="centered",
)


# =========================
# 基础工具函数
# =========================

def check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def run_command(command):
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def get_file_size_mb(uploaded_file) -> float:
    return uploaded_file.size / 1024 / 1024


def save_uploaded_file(uploaded_file, save_path: Path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in [
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
    ]


# =========================
# FFmpeg 处理函数
# =========================

def extract_audio_from_video(video_path: str, audio_path: str):
    """
    从视频中提取音频，并统一转成 16kHz 单声道 WAV。
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
    将长音频切分为多个 WAV 片段。
    每个片段再交给 Whisper 识别，避免一次性处理 1 小时音频导致内存爆掉。
    """
    chunks_path = Path(chunks_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)

    output_pattern = str(chunks_path / "chunk_%04d.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_pattern,
    ]

    run_command(command)

    return sorted(chunks_path.glob("chunk_*.wav"))


def get_media_duration_seconds(media_path: str) -> float | None:
    """
    使用 ffprobe 获取媒体时长。
    如果获取失败，返回 None，不影响主流程。
    """
    if shutil.which("ffprobe") is None:
        return None

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# =========================
# Whisper 识别函数
# =========================

@st.cache_resource
def load_model(model_size: str):
    """
    Streamlit Cloud 通常没有 GPU，因此强制使用 CPU + int8。
    这样更稳、更省内存。
    """
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
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


# =========================
# 输出文件生成
# =========================

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


# =========================
# 页面主体
# =========================

st.title("🎬 视频转文字 / 字幕工具")

st.write(
    "上传原视频后，系统会在云端自动完成：视频保存、音频提取、音频切片、AI 识别、TXT/SRT 生成。"
)

st.warning(
    "你选择的是全云端处理方案。1 小时视频可以尝试，但如果原视频过大，上传过程仍可能因网络或云端资源限制失败。"
)

if not check_ffmpeg_available():
    st.error(
        "当前环境未检测到 FFmpeg。请确认 GitHub 仓库根目录有 packages.txt，且内容为 ffmpeg。"
    )
    st.stop()


# =========================
# 侧边栏参数
# =========================

st.sidebar.header("识别参数")

model_size = st.sidebar.selectbox(
    "Whisper 模型",
    ["tiny", "base", "small", "medium"],
    index=2,
    help=(
        "Streamlit Cloud 免费环境建议 small。"
        "medium 可能更准，但 1 小时视频会更慢，也更容易内存不足。"
    ),
)

language_option = st.sidebar.selectbox(
    "语言",
    ["中文", "自动识别"],
    index=0,
)

language_map = {
    "中文": "zh",
    "自动识别": None,
}

language = language_map[language_option]

chunk_minutes = st.sidebar.selectbox(
    "音频切片长度",
    [2, 3, 5, 10],
    index=1,
    help="1 小时视频建议 3 分钟一段。机器压力更小，更稳。"
)

initial_prompt = st.sidebar.text_area(
    "贵州话识别提示词",
    value=(
        "这是贵州话、西南官话、贵州方言内容。"
        "请尽量识别为中文，保留说话原意。"
        "对于方言语气词和口语表达，请尽量转写成通顺中文。"
        "如果有地名、人名、村名、乡镇名，请尽量保留原文。"
    ),
    height=150,
)

show_video_preview = st.sidebar.checkbox(
    "显示视频预览",
    value=False,
    help="大视频预览可能增加页面负担，默认关闭。"
)

st.sidebar.markdown("---")
st.sidebar.write("推荐设置：")
st.sidebar.write("- 模型：small")
st.sidebar.write("- 语言：中文")
st.sidebar.write("- 切片：3 分钟")
st.sidebar.write("- 视频大小：尽量小于 1GB")


# =========================
# 上传视频
# =========================

uploaded_file = st.file_uploader(
    "上传原视频文件",
    type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
)

if uploaded_file is None:
    st.info("请上传一个视频文件。")
    st.stop()


file_size_mb = get_file_size_mb(uploaded_file)
file_stem = Path(uploaded_file.name).stem
uploaded_suffix = Path(uploaded_file.name).suffix.lower()

st.write(f"文件名：`{uploaded_file.name}`")
st.write(f"文件大小：`{file_size_mb:.2f} MB`")

if file_size_mb > 1000:
    st.error(
        "该视频超过 1000MB。虽然配置允许上传，但 Streamlit Cloud 免费环境很可能不稳定。"
    )
elif file_size_mb > 500:
    st.warning(
        "该视频超过 500MB，上传和处理可能较慢。如果失败，建议换更高配置云服务器部署。"
    )

if show_video_preview:
    with st.expander("视频预览", expanded=False):
        st.video(uploaded_file)

start_button = st.button("开始云端转文字", type="primary")


# =========================
# 主处理流程
# =========================

if start_button:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        video_path = temp_dir / uploaded_file.name
        extracted_audio_path = temp_dir / "extracted_audio.wav"
        chunks_dir = temp_dir / "chunks"

        try:
            progress = st.progress(0)
            status = st.empty()

            # 1. 保存上传视频
            status.write("步骤 1/5：正在保存上传的视频文件到云端临时目录...")
            progress.progress(5)

            save_uploaded_file(uploaded_file, video_path)

            if not is_video_file(video_path):
                st.error("不支持的视频格式。")
                st.stop()

            duration_seconds = get_media_duration_seconds(str(video_path))

            if duration_seconds:
                st.write(f"检测到视频时长：`{duration_seconds / 60:.1f} 分钟`")

            # 2. 提取音频
            status.write("步骤 2/5：正在从视频中提取音频...")
            progress.progress(15)

            extract_audio_from_video(
                str(video_path),
                str(extracted_audio_path),
            )

            # 3. 切分音频
            status.write("步骤 3/5：正在切分长音频...")
            progress.progress(30)

            chunk_seconds = chunk_minutes * 60

            chunk_paths = split_audio(
                str(extracted_audio_path),
                str(chunks_dir),
                chunk_seconds,
            )

            if not chunk_paths:
                st.error("音频切片失败，没有生成任何片段。")
                st.stop()

            total_chunks = len(chunk_paths)
            st.write(f"已切分音频片段数：`{total_chunks}`")

            # 4. 逐段识别
            all_segments = []
            detected_language = None
            detected_probability = None

            for index, chunk_path in enumerate(chunk_paths):
                status.write(
                    f"步骤 4/5：正在识别第 {index + 1} / {total_chunks} 段..."
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

                current_progress = 30 + int((index + 1) / total_chunks * 60)
                progress.progress(min(current_progress, 90))

            # 5. 生成结果
            status.write("步骤 5/5：正在生成 TXT 和 SRT 文件...")
            progress.progress(95)

            txt_content = make_txt(all_segments)
            txt_with_time_content = make_txt_with_time(all_segments)
            srt_content = make_srt(all_segments)

            progress.progress(100)
            status.write("处理完成。")

            st.success("转写完成")

            st.write(f"识别语言：`{detected_language}`")

            if detected_probability is not None:
                st.write(f"语言置信度：`{detected_probability:.2f}`")

            st.write(f"字幕段落数：`{len(all_segments)}`")

            st.subheader("转写预览")

            st.text_area(
                "带时间戳文本",
                txt_with_time_content,
                height=400,
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

        except MemoryError:
            st.error(
                "云端内存不足。建议把模型改为 tiny/base，或换更高配置服务器部署。"
            )

        except Exception as e:
            st.error("处理失败。")
            st.code(str(e))
