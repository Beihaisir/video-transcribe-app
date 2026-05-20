# 贵州话视频转文字 / 字幕工具

这是一个基于 Streamlit 和 faster-whisper 的全云端视频转文字工具。

## 功能

- 上传原视频
- 云端自动提取音频
- 云端自动切分长音频
- AI 识别贵州话 / 中文方言内容
- 生成 TXT 纯文本
- 生成 TXT 带时间戳
- 生成 SRT 字幕

## 项目结构

```text
video-transcribe-app/
├── app.py
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
└── README.md
