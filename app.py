import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="AI 视频/录音/链接转文字与会议纪要工具",
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


def save_uploaded_file(uploaded_file, save_path: Path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def get_file_size_mb(uploaded_file) -> float:
    return uploaded_file.size / 1024 / 1024


def safe_filename(name: str) -> str:
    cleaned = "".join(
        c for c in name if c.isalnum() or c in [" ", "_", "-", "."]
    ).strip()
    return cleaned or "output"


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


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


# ============================================================
# FFmpeg：音频处理
# ============================================================

def standardize_audio(input_path: str, output_path: str):
    """
    将视频或音频统一转为 16kHz 单声道 WAV。
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


def split_audio(audio_path: str, chunks_dir: str, chunk_seconds: int):
    """
    将长音频切片，避免一次性处理 1 小时内容导致内存不足。
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


# ============================================================
# 链接处理：yt-dlp
# ============================================================

def get_url_info(url: str) -> dict:
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
    仅处理平台公开允许解析的内容，不做去水印或绕过限制。
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


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


# ============================================================
# Whisper 转写
# ============================================================

@st.cache_resource
def load_whisper_model(model_size: str):
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
    model = load_whisper_model(model_size)

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
                "start": float(segment.start),
                "end": float(segment.end),
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
            audio_path=str(audio_path),
            chunks_dir=str(chunks_dir),
            chunk_seconds=chunk_seconds,
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
                audio_path=str(chunk_path),
                model_size=model_size,
                language=language,
                initial_prompt=initial_prompt,
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

            current_progress = 35 + int((index + 1) / total_chunks * 50)
            progress.progress(min(current_progress, 85))

        return all_segments, detected_language, detected_probability, total_chunks


# ============================================================
# 说话人分离
# ============================================================

@st.cache_resource
def load_diarization_pipeline(hf_token: str):
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    return pipeline


def diarize_audio(audio_path: str, hf_token: str):
    pipeline = load_diarization_pipeline(hf_token)
    diarization = pipeline(audio_path)

    speaker_turns = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })

    return speaker_turns


def overlap_seconds(a_start, a_end, b_start, b_end) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def assign_speakers_to_segments(segments, speaker_turns):
    enriched_segments = []

    for seg in segments:
        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap = overlap_seconds(
                seg["start"],
                seg["end"],
                turn["start"],
                turn["end"],
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        enriched_segments.append({
            **seg,
            "speaker": best_speaker,
        })

    return enriched_segments


def ensure_default_speaker(segments):
    output = []

    for item in segments:
        if "speaker" not in item:
            output.append({
                **item,
                "speaker": "SPEAKER_00",
            })
        else:
            output.append(item)

    return output


def get_unique_speakers(segments):
    speakers = []

    for item in segments:
        speaker = item.get("speaker", "SPEAKER_00")

        if speaker not in speakers:
            speakers.append(speaker)

    return speakers


def apply_speaker_name_map(segments, speaker_name_map):
    mapped = []

    for item in segments:
        raw_speaker = item.get("speaker", "SPEAKER_00")
        display_speaker = speaker_name_map.get(raw_speaker, raw_speaker)

        mapped.append({
            **item,
            "speaker": display_speaker,
        })

    return mapped


# ============================================================
# 文本 / 字幕输出
# ============================================================

def make_plain_txt(segments) -> str:
    return "\n".join(item["text"] for item in segments)


def make_speaker_txt(segments) -> str:
    lines = []

    for item in segments:
        speaker = item.get("speaker", "SPEAKER_00")
        lines.append(f"{speaker}：{item['text']}")

    return "\n".join(lines)


def make_speaker_txt_with_time(segments) -> str:
    lines = []

    for item in segments:
        start = format_timestamp(item["start"])
        end = format_timestamp(item["end"])
        speaker = item.get("speaker", "SPEAKER_00")
        lines.append(f"[{start} --> {end}] {speaker}：{item['text']}")

    return "\n".join(lines)


def make_speaker_srt(segments) -> str:
    lines = []

    for index, item in enumerate(segments, start=1):
        start = format_timestamp(item["start"])
        end = format_timestamp(item["end"])
        speaker = item.get("speaker", "SPEAKER_00")
        text = item["text"]

        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker}：{text}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# DeepSeek 总结 / 会议纪要
# ============================================================

def get_deepseek_client():
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY。请在 Streamlit Secrets 中添加。"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def build_minutes_prompt(transcript_text: str, meeting_title: str, content_type: str):
    return f"""
你是一名专业中文会议纪要和内容整理助理。请根据下面的分角色转写内容，生成结构清晰、适合保存、转发和复盘的纪要。

内容标题：{meeting_title}
内容类型：{content_type}

请严格输出 Markdown，结构如下：

# 会议纪要 / 内容纪要

## 一、基本信息
- 标题：
- 内容类型：
- 纪要生成说明：本纪要基于 AI 自动转写内容生成，需人工复核。

## 二、核心摘要
用 5 到 10 条 bullet 总结核心内容。

## 三、分角色观点整理
按说话人/角色分别整理其主要观点。
如果只有一个说话人，请按主题整理。

## 四、关键结论
列出已经形成的明确结论。
如果没有明确结论，请写“暂无明确结论”。

## 五、待办事项
请用 Markdown 表格输出：
| 事项 | 负责人 | 截止时间 | 备注 |
如果没有明确负责人或时间，请写“待确认”。

## 六、风险、问题与待确认事项
列出内容中提到的问题、分歧、风险或需要进一步确认的点。

## 七、重点原话摘录
摘录 5 到 10 条最重要的原话，保持简洁。

## 八、可对外发布的精简文案
输出一段更通顺、更适合对外发布、归档或二次创作的中文文案。

下面是转写内容：

{transcript_text}
""".strip()


def generate_ai_minutes(
    transcript_text: str,
    meeting_title: str,
    content_type: str,
    model_name: str,
):
    client = get_deepseek_client()

    prompt = build_minutes_prompt(
        transcript_text=transcript_text,
        meeting_title=meeting_title,
        content_type=content_type,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一名严谨、专业、擅长中文会议纪要和内容总结的助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ============================================================
# 渲染结果
# ============================================================

def render_platform_metadata(result):
    title = result.get("title", "")
    description = result.get("description", "")
    thumbnail = result.get("thumbnail", "")
    platform_subtitles = result.get("platform_subtitles", [])

    if title:
        st.subheader("链接信息")
        st.write(f"标题：`{title}`")

    if thumbnail:
        st.image(thumbnail, caption="封面预览")

    if description:
        with st.expander("平台描述 / 文案", expanded=False):
            st.write(description)

        st.download_button(
            label="下载平台描述文案 TXT",
            data=description,
            file_name=f"{safe_filename(title or 'description')}_description.txt",
            mime="text/plain",
        )

    if platform_subtitles:
        st.subheader("平台字幕 / 自动字幕")

        for idx, subtitle in enumerate(platform_subtitles, start=1):
            st.download_button(
                label=f"下载平台字幕 {idx}",
                data=subtitle["content"],
                file_name=subtitle["filename"],
                mime="text/plain",
            )


def render_transcription_result():
    result = st.session_state.get("transcription_result")

    if not result:
        return

    file_stem = result["file_stem"]
    raw_segments = result["segments"]

    st.markdown("---")
    st.header("转写结果")

    render_platform_metadata(result)

    speakers = get_unique_speakers(raw_segments)

    st.subheader("角色命名")

    speaker_name_map = {}

    for speaker in speakers:
        default_value = st.session_state.get(
            f"speaker_name_{speaker}",
            speaker,
        )

        speaker_name_map[speaker] = st.text_input(
            f"{speaker} 显示为",
            value=default_value,
            key=f"speaker_name_{speaker}",
            help="例如：主持人、客户、嘉宾、老师、学生、负责人等。",
        )

    mapped_segments = apply_speaker_name_map(
        raw_segments,
        speaker_name_map,
    )

    txt_content = make_speaker_txt(mapped_segments)
    txt_with_time_content = make_speaker_txt_with_time(mapped_segments)
    srt_content = make_speaker_srt(mapped_segments)

    st.subheader("带时间戳预览")

    st.text_area(
        "转写文本",
        txt_with_time_content,
        height=420,
    )

    st.subheader("下载转写文件")

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

    st.markdown("---")
    st.header("DeepSeek AI 总结 / 会议纪要")

    content_type = st.selectbox(
        "内容类型",
        ["会议", "访谈", "课程", "培训", "短视频素材", "普通录音", "其他"],
        index=0,
    )

    meeting_title = st.text_input(
        "纪要标题",
        value=result.get("title") or file_stem,
    )

    summary_model = st.selectbox(
        "DeepSeek 模型",
        ["deepseek-chat", "deepseek-reasoner"],
        index=0,
        help="deepseek-chat 速度较快；deepseek-reasoner 更适合复杂推理，但可能更慢。",
    )

    if st.button("生成 AI 总结和会议纪要", type="primary"):
        try:
            with st.spinner("DeepSeek 正在生成会议纪要..."):
                minutes_markdown = generate_ai_minutes(
                    transcript_text=txt_with_time_content,
                    meeting_title=meeting_title,
                    content_type=content_type,
                    model_name=summary_model,
                )

            st.session_state["minutes_markdown"] = minutes_markdown
            st.session_state["minutes_file_stem"] = file_stem

        except Exception as e:
            st.error("AI 会议纪要生成失败。请检查 DEEPSEEK_API_KEY、余额、网络或模型名称。")
            st.code(str(e))

    minutes_markdown = st.session_state.get("minutes_markdown")

    if minutes_markdown:
        st.subheader("AI 会议纪要")
        st.markdown(minutes_markdown)

        minutes_file_stem = st.session_state.get("minutes_file_stem", file_stem)

        st.download_button(
            label="下载会议纪要 Markdown",
            data=minutes_markdown,
            file_name=f"{minutes_file_stem}_meeting_minutes.md",
            mime="text/markdown",
        )

        st.download_button(
            label="下载会议纪要 TXT",
            data=minutes_markdown,
            file_name=f"{minutes_file_stem}_meeting_minutes.txt",
            mime="text/plain",
        )


# ============================================================
# 主处理流程
# ============================================================

def process_audio_to_segments(
    source_path: Path,
    standard_audio_path: Path,
    file_stem: str,
    title: str,
    description: str,
    thumbnail: str,
    platform_subtitles: list[dict],
    model_size: str,
    language,
    initial_prompt: str,
    chunk_minutes: int,
    enable_diarization: bool,
    progress,
    status,
):
    status.write("正在标准化音频...")
    progress.progress(25)

    standardize_audio(
        input_path=str(source_path),
        output_path=str(standard_audio_path),
    )

    duration = get_media_duration_seconds(str(standard_audio_path))

    if duration:
        st.write(f"检测到音频时长：`{duration / 60:.1f} 分钟`")

    segments, detected_language, detected_probability, total_chunks = transcribe_audio_by_chunks(
        audio_path=str(standard_audio_path),
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
        chunk_minutes=chunk_minutes,
        progress=progress,
        status=status,
    )

    if enable_diarization:
        hf_token = st.secrets.get("HF_TOKEN", "")

        if not hf_token:
            st.warning("未配置 HF_TOKEN，已跳过说话人分离。")
            segments = ensure_default_speaker(segments)
        else:
            try:
                status.write("正在进行说话人分离...")
                progress.progress(88)

                speaker_turns = diarize_audio(
                    audio_path=str(standard_audio_path),
                    hf_token=hf_token,
                )

                segments = assign_speakers_to_segments(
                    segments,
                    speaker_turns,
                )

            except Exception as e:
                st.warning("说话人分离失败，已保留普通转写结果。")
                st.code(str(e))
                segments = ensure_default_speaker(segments)
    else:
        segments = ensure_default_speaker(segments)

    progress.progress(100)
    status.write("处理完成。")

    st.session_state["transcription_result"] = {
        "file_stem": safe_filename(file_stem),
        "title": title,
        "description": description,
        "thumbnail": thumbnail,
        "platform_subtitles": platform_subtitles,
        "segments": segments,
        "detected_language": detected_language,
        "detected_probability": detected_probability,
        "total_chunks": total_chunks,
    }

    st.session_state.pop("minutes_markdown", None)
    st.session_state.pop("minutes_file_stem", None)

    st.success("转写完成")

    st.write(f"识别语言：`{detected_language}`")

    if detected_probability is not None:
        st.write(f"语言置信度：`{detected_probability:.2f}`")

    st.write(f"音频切片数：`{total_chunks}`")
    st.write(f"字幕段落数：`{len(segments)}`")


# ============================================================
# 页面主体
# ============================================================

st.title("🎬 AI 视频 / 录音 / 链接转文字与会议纪要工具")

st.write(
    "支持上传视频、上传录音、粘贴公开视频链接，自动转写、分角色、生成 DeepSeek AI 总结和会议纪要。"
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

st.sidebar.header("转写参数")

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
    help="1 小时内容建议 3 分钟一段。",
)

initial_prompt = st.sidebar.text_area(
    "识别提示词",
    value=(
        "这是中文内容，可能包含贵州话、西南官话、贵州方言。"
        "请尽量识别为中文，保留说话原意。"
        "对于方言语气词和口语表达，请尽量转写成通顺中文。"
        "如果有地名、人名、村名、乡镇名，请尽量保留原文。"
    ),
    height=160,
)

st.sidebar.header("角色与纪要")

enable_diarization = st.sidebar.checkbox(
    "启用自动说话人分离",
    value=False,
    help="需要配置 HF_TOKEN。会明显增加处理时间和部署资源压力。",
)

st.sidebar.info(
    "即使不启用自动说话人分离，也可以在转写完成后手动把 SPEAKER_00 改成主持人、嘉宾等角色。"
)


# ============================================================
# 输入模式
# ============================================================

input_mode = st.radio(
    "选择输入方式",
    [
        "上传视频",
        "上传录音/音频",
        "粘贴公开视频链接",
    ],
    horizontal=False,
)


# ============================================================
# 模式一：上传视频
# ============================================================

if input_mode == "上传视频":
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
                standard_audio_path = temp_dir / "standard_audio.wav"

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("正在保存上传视频...")
                    progress.progress(10)

                    save_uploaded_file(uploaded_file, input_path)

                    duration = get_media_duration_seconds(str(input_path))

                    if duration:
                        st.write(f"检测到视频时长：`{duration / 60:.1f} 分钟`")

                    process_audio_to_segments(
                        source_path=input_path,
                        standard_audio_path=standard_audio_path,
                        file_stem=file_stem,
                        title=file_stem,
                        description="",
                        thumbnail="",
                        platform_subtitles=[],
                        model_size=model_size,
                        language=language,
                        initial_prompt=initial_prompt,
                        chunk_minutes=chunk_minutes,
                        enable_diarization=enable_diarization,
                        progress=progress,
                        status=status,
                    )

                except subprocess.CalledProcessError as e:
                    st.error("视频或音频处理失败。")
                    st.code(e.stderr or str(e))
                except Exception as e:
                    st.error("处理失败。")
                    st.code(str(e))


# ============================================================
# 模式二：上传录音 / 音频
# ============================================================

elif input_mode == "上传录音/音频":
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
                standard_audio_path = temp_dir / "standard_audio.wav"

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("正在保存上传音频...")
                    progress.progress(10)

                    save_uploaded_file(uploaded_file, input_path)

                    duration = get_media_duration_seconds(str(input_path))

                    if duration:
                        st.write(f"检测到音频时长：`{duration / 60:.1f} 分钟`")

                    process_audio_to_segments(
                        source_path=input_path,
                        standard_audio_path=standard_audio_path,
                        file_stem=file_stem,
                        title=file_stem,
                        description="",
                        thumbnail="",
                        platform_subtitles=[],
                        model_size=model_size,
                        language=language,
                        initial_prompt=initial_prompt,
                        chunk_minutes=chunk_minutes,
                        enable_diarization=enable_diarization,
                        progress=progress,
                        status=status,
                    )

                except subprocess.CalledProcessError as e:
                    st.error("音频处理失败。")
                    st.code(e.stderr or str(e))
                except Exception as e:
                    st.error("处理失败。")
                    st.code(str(e))


# ============================================================
# 模式三：公开视频链接
# ============================================================

elif input_mode == "粘贴公开视频链接":
    url = st.text_input(
        "粘贴公开视频分享链接",
        placeholder="https://...",
    )

    prefer_platform_subtitle = st.checkbox(
        "优先尝试提取平台字幕",
        value=True,
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
                    file_stem = safe_filename(title)

                    platform_subtitles = []

                    if prefer_platform_subtitle:
                        status.write("正在尝试提取平台字幕...")
                        progress.progress(12)

                        subtitle_files = download_subtitles_from_url(
                            url=url,
                            output_dir=str(temp_dir),
                        )

                        for idx, subtitle_file in enumerate(subtitle_files, start=1):
                            platform_subtitles.append({
                                "filename": f"{file_stem}_platform_subtitle_{idx}.srt",
                                "content": read_text_file(subtitle_file),
                            })

                    status.write("正在从链接提取音频...")
                    progress.progress(18)

                    linked_audio_path = download_audio_from_url(
                        url=url,
                        output_dir=str(temp_dir),
                    )

                    standard_audio_path = temp_dir / "standard_audio.wav"

                    process_audio_to_segments(
                        source_path=linked_audio_path,
                        standard_audio_path=standard_audio_path,
                        file_stem=file_stem,
                        title=title,
                        description=description,
                        thumbnail=thumbnail,
                        platform_subtitles=platform_subtitles,
                        model_size=model_size,
                        language=language,
                        initial_prompt=initial_prompt,
                        chunk_minutes=chunk_minutes,
                        enable_diarization=enable_diarization,
                        progress=progress,
                        status=status,
                    )

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
# 渲染已生成结果
# ============================================================

render_transcription_result()
