# AI 视频 / 录音 / 链接转文字与会议纪要工具

这是一个基于 Streamlit、FFmpeg、faster-whisper、yt-dlp 和 DeepSeek API 的内容整理工具。

## 功能

- 上传视频转文字
- 上传录音/音频转文字
- 粘贴公开视频链接转文字
- 提取公开视频标题、描述、封面、字幕
- 可选说话人分离
- 手动角色命名
- DeepSeek AI 总结
- 自动生成会议纪要
- 下载 TXT、带时间戳 TXT、SRT、Markdown 纪要

## 合规说明

本工具仅用于处理用户有权处理的内容。

不支持：

- 去水印
- 无水印解析
- 绕过登录限制
- 绕过会员限制
- 绕过 DRM
- 绕过签名或平台下载限制
- 处理私密或未授权内容

## 项目结构

```text
video-transcribe-app/
├── app.py
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
└── README.md
