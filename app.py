import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel
from PIL import Image


st.set_page_config(
    page_title="AI 视频/录音/素材处理工具",
    page_icon="🎬",
    layout="centered",
)


# ============================================================
# 基础工具
# ============================================================

def check_command_available(command_name: str) -> bool:
    return shutil.which(command_name) is not None


def run_command(command):
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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


def safe_filename(name: str) -> str:
    cleaned = "".join(
        c for c in name if c.isalnum() or c in [" ", "_", "-", "."]
    ).strip()
    return cleaned or "output"


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in [
        ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"
    ]


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in [
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
    ]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    ]


# ============================================================
# FFmpeg：音视频处理
# ============================================================

def standardize_audio(input_path: str, output_path: str):
    """
    将音频或视频统一转为 16kHz 单声道 WAV。
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    run_command(command)


def extract_audio_mp3(input_video_path: str, output_audio_path: str):
    """
    从视频提取 MP3 音频，方便下载归档。
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        output_audio_path,
    ]
    run_command(command)


def split_audio(audio_path: str, chunks_dir: str, chunk_seconds: int):
    """
    将长音频切片，避免一次性识别长音频导致内存不足。
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
    if not check_command_available("ffprobe"):
        return None

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]

    try:
        result = run_command(command)
        return float(result.stdout.strip())
    except Exception:
        return None


def capture_video_frame(
    video_path: str,
    output_image_path: str,
    timestamp_seconds: int,
):
    """
    从视频指定秒数截图。
    """
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp_seconds),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_image_path,
    ]
    run_command(command)


def compress_video(
    input_video_path: str,
    output_video_path: str,
    crf: int,
    width: int | None,
):
    """
    压缩视频。crf 越大，体积越小，画质越低。
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_video_path,
    ]

    video_filters = []

    if width:
        video_filters.append(f"scale={width}:-2")

    if video_filters:
        command += ["-vf", ",".join(video_filters)]

    command += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "96k",
        output_video_path,
    ]

    run_command(command)


# ============================================================
# 图片处理
# ============================================================

def process_image(
    input_image_path: str,
    output_image_path: str,
    output_format: str,
    max_width: int | None,
    quality: int,
):
    image = Image.open(input_image_path)

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    if max_width and image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height))

    save_kwargs = {}

    if output_format.upper() in ["JPEG", "WEBP"]:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    image.save(output_image_path, output_format.upper(), **save_kwargs)


# ============================================================
# 链接处理：yt-dlp
# ============================================================

def get_url_info(url: str) -> dict:
    """
    获取公开链接信息。
    """
    command = [
        "yt-dlp",
        "--dump-json",
        "--skip-download",
        "--no-playlist",
        url,
    ]

    result = run_command(command)
    return json.loads(result.stdout)


def download_audio_from_url(url: str, output_dir: str) -> Path:
    """
    从公开视频链接提取音频。
    不绕过登录、会员、DRM、私密权限或平台限制。
    """
    output_template = str(Path(output_dir) / "source_audio.%(ext)s")

    command = [
        "yt-dlp",
        "--no-playlist",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", output_template,
        url,
    ]

    run_command(command)

    candidates = list(Path(output_dir).glob("source_audio.*"))

    if not candidates:
        raise FileNotFoundError("未能从链接中提取音频。")

    return candidates[0]


def download_subtitles_from_url(url: str, output_dir: str) -> list[Path]:
    """
    尝试下载平台自带字幕或自动字幕。
    """
    output_template = str(Path(output_dir) / "subtitle.%(ext)s")

    command = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "zh.*,en.*,all",
        "--convert-subs", "srt",
        "--no-playlist",
        "-o", output_template,
        url,
    ]

    try:
        run_command(command)
    except subprocess.CalledProcessError:
        return []

    return sorted(Path(output_dir).glob("subtitle*.srt"))


def download_public_media_if_allowed(url: str, output_dir: str) -> Path:
    """
    下载平台正常公开允许的媒体文件。
    不去水印，不绕过登录、会员、DRM、签名或私密限制。
    """
    output_template = str(Path(output_dir) / "public_media.%(ext)s")

    command = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[ext=mp4]/best",
        "-o", output_template,
        url,
    ]

    run_command(command)

    candidates = list(Path(output_dir).glob("public_media.*"))

    if not candidates:
        raise FileNotFoundError("未能下载平台公开允许的媒体文件。")

    return candidates[0]


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


# ============================================================
# Whisper 识别
# ============================================================

@st.cache_resource
def load_model(model_size: str):
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


def transcribe_audio_by_chunks(
    audio_path: str,
    model_size: str,
    language,
    initial_prompt: str | None,
    chunk_minutes: int,
    progress,
    status,
):
    chunk_seconds = chunk_minutes * 60

    with tempfile.TemporaryDirectory() as chunk_temp_dir:
        chunks_dir = Path(chunk_temp_dir) / "chunks"

        status.write("正在切分音频...")
        progress.progress(35)

        chunk_paths = split_audio(
            str(audio_path),
            str(chunks_dir),
            chunk_seconds,
        )

        if not chunk_paths:
            raise RuntimeError("音频切片失败，没有生成任何片段。")

        total_chunks = len(chunk_paths)

        all_segments = []
        detected_language = None
        detected_probability = None

        for index, chunk_path in enumerate(chunk_paths):
            status.write(f"正在识别第 {index + 1} / {total_chunks} 段...")

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

            current_progress = 35 + int((index + 1) / total_chunks * 55)
            progress.progress(min(current_progress, 90))

        return all_segments, detected_language, detected_probability, total_chunks


# ============================================================
# 文本输出
# ============================================================

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


def render_transcription_downloads(file_stem: str, segments):
    txt_content = make_txt(segments)
    txt_with_time_content = make_txt_with_time(segments)
    srt_content = make_srt(segments)

    st.subheader("转写预览")

    st.text_area(
        "带时间戳文本",
        txt_with_time_content,
        height=400,
    )

    st.subheader("下载转写结果")

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


def render_subtitle_downloads(file_stem: str, subtitle_files: list[Path]):
    if not subtitle_files:
        return

    st.subheader("平台字幕 / 自动字幕")

    for idx, subtitle_file in enumerate(subtitle_files, start=1):
        content = read_text_file(subtitle_file)

        st.download_button(
            label=f"下载平台字幕 {idx}",
            data=content,
            file_name=f"{file_stem}_platform_subtitle_{idx}.srt",
            mime="text/plain",
        )


# ============================================================
# 页面主体
# ============================================================

st.title("🎬 AI 视频 / 录音 / 链接 / 自有素材处理工具")

st.write(
    "支持视频转文字、录音转文字、公开视频链接转文字，以及自有原始素材处理。"
)

st.warning(
    "本工具仅处理你有权处理的内容。不提供去水印、无水印解析、绕过登录、会员、DRM、签名或平台限制的功能。"
)

if not check_command_available("ffmpeg"):
    st.error("当前环境未检测到 FFmpeg。请确认 packages.txt 中包含 ffmpeg。")
    st.stop()

if not check_command_available("yt-dlp"):
    st.error("当前环境未检测到 yt-dlp。请确认 requirements.txt 中包含 yt-dlp。")
    st.stop()


# ============================================================
# 侧边栏参数
# ============================================================

st.sidebar.header("AI 识别参数")

model_size = st.sidebar.selectbox(
    "Whisper 模型",
    ["tiny", "base", "small", "medium"],
    index=2,
    help="Streamlit Cloud 免费环境建议 small。medium 更准但更慢。",
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
    "切片长度",
    [2, 3, 5, 10],
    index=1,
    help="长视频/长录音建议 3 分钟一段。",
)

initial_prompt = st.sidebar.text_area(
    "识别提示词",
    value=(
        "这是中文内容，可能包含贵州话、西南官话、贵州方言。"
        "请尽量识别为中文，保留说话原意。"
        "对于方言语气词和口语表达，请尽量转写成通顺中文。"
        "如果有地名、人名、村名、乡镇名，请尽量保留原文。"
    ),
    height=150,
)

st.sidebar.markdown("---")
st.sidebar.write("推荐：")
st.sidebar.write("- 1 小时内容：small + 3 分钟切片")
st.sidebar.write("- 贵州话：语言选择中文")
st.sidebar.write("- 大视频：关闭预览")


# ============================================================
# 输入模式
# ============================================================

input_mode = st.radio(
    "选择功能",
    [
        "上传视频转文字",
        "上传录音/音频转文字",
        "粘贴公开视频链接转文字",
        "自有原始视频处理",
        "自有原始图片处理",
    ],
    horizontal=False,
)


# ============================================================
# 模式 1：上传视频转文字
# ============================================================

if input_mode == "上传视频转文字":
    uploaded_file = st.file_uploader(
        "上传视频文件",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
    )

    if uploaded_file is not None:
        file_size_mb = get_file_size_mb(uploaded_file)
        file_stem = Path(uploaded_file.name).stem

        st.write(f"文件名：`{uploaded_file.name}`")
        st.write(f"文件大小：`{file_size_mb:.2f} MB`")

        if file_size_mb > 500:
            st.warning("视频超过 500MB，Streamlit Cloud 免费环境可能不稳定。")

        show_preview = st.checkbox("显示视频预览", value=False)

        if show_preview:
            st.video(uploaded_file)

        if st.button("开始视频转文字", type="primary"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                input_path = temp_dir / uploaded_file.name
                audio_path = temp_dir / "standard_audio.wav"

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("正在保存上传视频...")
                    progress.progress(10)
                    save_uploaded_file(uploaded_file, input_path)

                    duration = get_media_duration_seconds(str(input_path))
                    if duration:
                        st.write(f"检测到视频时长：`{duration / 60:.1f} 分钟`")

                    status.write("正在提取并标准化音频...")
                    progress.progress(25)
                    standardize_audio(str(input_path), str(audio_path))

                    segments, detected_language, detected_probability, total_chunks = transcribe_audio_by_chunks(
                        audio_path=audio_path,
                        model_size=model_size,
                        language=language,
                        initial_prompt=initial_prompt,
                        chunk_minutes=chunk_minutes,
                        progress=progress,
                        status=status,
                    )

                    progress.progress(100)
                    status.write("处理完成。")

                    st.success("视频转文字完成")
                    st.write(f"识别语言：`{detected_language}`")

                    if detected_probability is not None:
                        st.write(f"语言置信度：`{detected_probability:.2f}`")

                    st.write(f"音频切片数：`{total_chunks}`")
                    st.write(f"字幕段落数：`{len(segments)}`")

                    render_transcription_downloads(file_stem, segments)

                except subprocess.CalledProcessError as e:
                    st.error("FFmpeg 处理失败。")
                    st.code(e.stderr or str(e))
                except Exception as e:
                    st.error("处理失败。")
                    st.code(str(e))


# ============================================================
# 模式 2：上传录音/音频转文字
# ============================================================

elif input_mode == "上传录音/音频转文字":
    uploaded_file = st.file_uploader(
        "上传录音或音频文件",
        type=["mp3", "wav", "m4a", "flac", "aac", "ogg", "wma"],
    )

    if uploaded_file is not None:
        file_size_mb = get_file_size_mb(uploaded_file)
        file_stem = Path(uploaded_file.name).stem

        st.write(f"文件名：`{uploaded_file.name}`")
        st.write(f"文件大小：`{file_size_mb:.2f} MB`")

        st.audio(uploaded_file)

        if st.button("开始录音转文字", type="primary"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                input_path = temp_dir / uploaded_file.name
                audio_path = temp_dir / "standard_audio.wav"

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("正在保存上传音频...")
                    progress.progress(10)
                    save_uploaded_file(uploaded_file, input_path)

                    duration = get_media_duration_seconds(str(input_path))
                    if duration:
                        st.write(f"检测到音频时长：`{duration / 60:.1f} 分钟`")

                    status.write("正在标准化音频...")
                    progress.progress(25)
                    standardize_audio(str(input_path), str(audio_path))

                    segments, detected_language, detected_probability, total_chunks = transcribe_audio_by_chunks(
                        audio_path=audio_path,
                        model_size=model_size,
                        language=language,
                        initial_prompt=initial_prompt,
                        chunk_minutes=chunk_minutes,
                        progress=progress,
                        status=status,
                    )

                    progress.progress(100)
                    status.write("处理完成。")

                    st.success("录音转文字完成")
                    st.write(f"识别语言：`{detected_language}`")

                    if detected_probability is not None:
                        st.write(f"语言置信度：`{detected_probability:.2f}`")

                    st.write(f"音频切片数：`{total_chunks}`")
                    st.write(f"字幕段落数：`{len(segments)}`")

                    render_transcription_downloads(file_stem, segments)

                except subprocess.CalledProcessError as e:
                    st.error("音频处理失败。")
                    st.code(e.stderr or str(e))
                except Exception as e:
                    st.error("处理失败。")
                    st.code(str(e))


# ============================================================
# 模式 3：公开视频链接转文字
# ============================================================

elif input_mode == "粘贴公开视频链接转文字":
    url = st.text_input(
        "粘贴公开视频分享链接",
        placeholder="https://...",
    )

    prefer_platform_subtitle = st.checkbox(
        "优先尝试提取平台字幕",
        value=True,
    )

    download_public_media = st.checkbox(
        "同时下载平台公开允许的媒体文件",
        value=False,
        help="不会去水印，不绕过登录、会员、DRM、签名或平台限制。",
    )

    st.info(
        "链接模式仅支持公开视频，且需要平台允许 yt-dlp 正常解析。"
        "私密、登录后可见、会员、DRM 或平台限制内容可能失败。"
    )

    if url:
        if st.button("分析链接并转文字", type="primary"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("正在读取链接信息...")
                    progress.progress(5)

                    info = get_url_info(url)

                    title = info.get("title") or "linked_video"
                    description = info.get("description") or ""
                    thumbnail = info.get("thumbnail") or ""

                    safe_title = safe_filename(title)

                    st.subheader("链接信息")
                    st.write(f"标题：`{title}`")

                    if thumbnail:
                        st.image(thumbnail, caption="封面预览")

                    if description:
                        with st.expander("查看平台描述/文案", expanded=False):
                            st.write(description)

                        st.download_button(
                            label="下载平台描述文案 TXT",
                            data=description,
                            file_name=f"{safe_title}_description.txt",
                            mime="text/plain",
                        )

                    subtitle_files = []

                    if prefer_platform_subtitle:
                        status.write("正在尝试提取平台字幕...")
                        progress.progress(15)

                        subtitle_files = download_subtitles_from_url(
                            url,
                            str(temp_dir),
                        )

                        if subtitle_files:
                            st.success("已检测到平台字幕/自动字幕。")
                            render_subtitle_downloads(safe_title, subtitle_files)
                        else:
                            st.info("未提取到可用平台字幕，将继续提取音频并 AI 转写。")

                    if download_public_media:
                        status.write("正在下载平台公开允许的媒体文件...")
                        progress.progress(20)

                        public_media_path = download_public_media_if_allowed(
                            url,
                            str(temp_dir),
                        )

                        st.download_button(
                            label="下载平台公开允许的媒体文件",
                            data=public_media_path.read_bytes(),
                            file_name=public_media_path.name,
                            mime="application/octet-stream",
                        )

                    status.write("正在从链接提取音频...")
                    progress.progress(25)

                    linked_audio_path = download_audio_from_url(
                        url,
                        str(temp_dir),
                    )

                    standard_audio_path = temp_dir / "standard_audio.wav"

                    status.write("正在标准化音频...")
                    progress.progress(30)

                    standardize_audio(
                        str(linked_audio_path),
                        str(standard_audio_path),
                    )

                    duration = get_media_duration_seconds(str(standard_audio_path))

                    if duration:
                        st.write(f"检测到音频时长：`{duration / 60:.1f} 分钟`")

                    segments, detected_language, detected_probability, total_chunks = transcribe_audio_by_chunks(
                        audio_path=standard_audio_path,
                        model_size=model_size,
                        language=language,
                        initial_prompt=initial_prompt,
                        chunk_minutes=chunk_minutes,
                        progress=progress,
                        status=status,
                    )

                    progress.progress(100)
                    status.write("处理完成。")

                    st.success("链接转文字完成")
                    st.write(f"识别语言：`{detected_language}`")

                    if detected_probability is not None:
                        st.write(f"语言置信度：`{detected_probability:.2f}`")

                    st.write(f"音频切片数：`{total_chunks}`")
                    st.write(f"字幕段落数：`{len(segments)}`")

                    render_transcription_downloads(safe_title, segments)

                except subprocess.CalledProcessError as e:
                    st.error("链接解析或音频提取失败。")
                    st.write(
                        "可能原因：平台不支持、需要登录、链接不是公开视频、内容受保护、平台临时限制。"
                    )
                    st.code(e.stderr or str(e))
                except Exception as e:
                    st.error("处理失败。")
                    st.code(str(e))


# ============================================================
# 模式 4：自有原始视频处理
# ============================================================

elif input_mode == "自有原始视频处理":
    st.write(
        "用于处理你自己上传的原始视频素材：提取音频、截图封面、压缩视频、转文字。"
    )

    uploaded_file = st.file_uploader(
        "上传自有原始视频",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
    )

    if uploaded_file is not None:
        file_size_mb = get_file_size_mb(uploaded_file)
        file_stem = Path(uploaded_file.name).stem

        st.write(f"文件名：`{uploaded_file.name}`")
        st.write(f"文件大小：`{file_size_mb:.2f} MB`")

        with st.expander("视频处理选项", expanded=True):
            do_transcribe = st.checkbox("生成文字稿和字幕", value=True)
            do_extract_audio = st.checkbox("提取 MP3 音频", value=True)
            do_capture_cover = st.checkbox("生成视频封面截图", value=True)
            do_compress_video = st.checkbox("压缩视频", value=False)

            cover_second = st.number_input(
                "封面截图时间点：第几秒",
                min_value=0,
                value=3,
                step=1,
            )

            crf = st.slider(
                "视频压缩质量 CRF，数值越大体积越小",
                min_value=18,
                max_value=35,
                value=28,
            )

            width_option = st.selectbox(
                "压缩视频宽度",
                ["保持原始宽度", "1920", "1280", "720"],
                index=2,
            )

        show_preview = st.checkbox("显示视频预览", value=False)

        if show_preview:
            st.video(uploaded_file)

        if st.button("开始处理自有视频", type="primary"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                input_path = temp_dir / uploaded_file.name

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("正在保存视频...")
                    progress.progress(5)
                    save_uploaded_file(uploaded_file, input_path)

                    duration = get_media_duration_seconds(str(input_path))

                    if duration:
                        st.write(f"检测到视频时长：`{duration / 60:.1f} 分钟`")

                    progress.progress(15)

                    if do_extract_audio:
                        status.write("正在提取 MP3 音频...")
                        audio_mp3_path = temp_dir / f"{file_stem}_audio.mp3"

                        extract_audio_mp3(
                            str(input_path),
                            str(audio_mp3_path),
                        )

                        st.download_button(
                            label="下载提取的 MP3 音频",
                            data=audio_mp3_path.read_bytes(),
                            file_name=f"{file_stem}_audio.mp3",
                            mime="audio/mpeg",
                        )

                    progress.progress(30)

                    if do_capture_cover:
                        status.write("正在生成封面截图...")
                        cover_path = temp_dir / f"{file_stem}_cover.jpg"

                        capture_video_frame(
                            str(input_path),
                            str(cover_path),
                            int(cover_second),
                        )

                        st.image(str(cover_path), caption="封面截图")

                        st.download_button(
                            label="下载封面截图",
                            data=cover_path.read_bytes(),
                            file_name=f"{file_stem}_cover.jpg",
                            mime="image/jpeg",
                        )

                    progress.progress(45)

                    if do_compress_video:
                        status.write("正在压缩视频...")

                        compressed_path = temp_dir / f"{file_stem}_compressed.mp4"

                        width = None if width_option == "保持原始宽度" else int(width_option)

                        compress_video(
                            str(input_path),
                            str(compressed_path),
                            crf=crf,
                            width=width,
                        )

                        st.download_button(
                            label="下载压缩后视频",
                            data=compressed_path.read_bytes(),
                            file_name=f"{file_stem}_compressed.mp4",
                            mime="video/mp4",
                        )

                    progress.progress(60)

                    if do_transcribe:
                        status.write("正在提取并标准化音频用于识别...")
                        standard_audio_path = temp_dir / "standard_audio.wav"

                        standardize_audio(
                            str(input_path),
                            str(standard_audio_path),
                        )

                        segments, detected_language, detected_probability, total_chunks = transcribe_audio_by_chunks(
                            audio_path=standard_audio_path,
                            model_size=model_size,
                            language=language,
                            initial_prompt=initial_prompt,
                            chunk_minutes=chunk_minutes,
                            progress=progress,
                            status=status,
                        )

                        st.success("视频转文字完成")
                        st.write(f"识别语言：`{detected_language}`")

                        if detected_probability is not None:
                            st.write(f"语言置信度：`{detected_probability:.2f}`")

                        st.write(f"音频切片数：`{total_chunks}`")
                        st.write(f"字幕段落数：`{len(segments)}`")

                        render_transcription_downloads(file_stem, segments)

                    progress.progress(100)
                    status.write("处理完成。")

                except subprocess.CalledProcessError as e:
                    st.error("视频处理失败。")
                    st.code(e.stderr or str(e))
                except Exception as e:
                    st.error("处理失败。")
                    st.code(str(e))


# ============================================================
# 模式 5：自有原始图片处理
# ============================================================

elif input_mode == "自有原始图片处理":
    st.write(
        "用于处理你自己上传的原始图片素材：压缩、缩放、格式转换。"
    )

    uploaded_file = st.file_uploader(
        "上传自有原始图片",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded_file is not None:
        file_size_mb = get_file_size_mb(uploaded_file)
        file_stem = Path(uploaded_file.name).stem

        st.write(f"文件名：`{uploaded_file.name}`")
        st.write(f"文件大小：`{file_size_mb:.2f} MB`")

        st.image(uploaded_file, caption="原始图片预览")

        output_format = st.selectbox(
            "输出格式",
            ["JPEG", "PNG", "WEBP"],
            index=0,
        )

        max_width = st.selectbox(
            "最大宽度",
            ["保持原始宽度", "1920", "1280", "1080", "720"],
            index=1,
        )

        quality = st.slider(
            "图片质量",
            min_value=40,
            max_value=100,
            value=85,
        )

        if st.button("开始处理图片", type="primary"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                input_path = temp_dir / uploaded_file.name

                extension_map = {
                    "JPEG": "jpg",
                    "PNG": "png",
                    "WEBP": "webp",
                }

                output_ext = extension_map[output_format]
                output_path = temp_dir / f"{file_stem}_processed.{output_ext}"

                try:
                    save_uploaded_file(uploaded_file, input_path)

                    width = None if max_width == "保持原始宽度" else int(max_width)

                    process_image(
                        input_image_path=str(input_path),
                        output_image_path=str(output_path),
                        output_format=output_format,
                        max_width=width,
                        quality=quality,
                    )

                    st.success("图片处理完成")

                    st.image(str(output_path), caption="处理后图片预览")

                    mime_map = {
                        "JPEG": "image/jpeg",
                        "PNG": "image/png",
                        "WEBP": "image/webp",
                    }

                    st.download_button(
                        label="下载处理后图片",
                        data=output_path.read_bytes(),
                        file_name=output_path.name,
                        mime=mime_map[output_format],
                    )

                except Exception as e:
                    st.error("图片处理失败。")
                    st.code(str(e))
