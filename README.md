# 音频分割与转录工具

AudioToSrt是一个用于将音频文件转录成文本并生成SRT字幕文件的工具。
它在本地加载开源模型并在本地执行。

## 功能

1. **音频分割**：根据停顿分割音频，并确保每个片段不超过30秒。
2. **语音转文字**：使用Hugging Face的Whisper模型批量转录音频片段。
3. **生成SRT文件**：根据转录结果生成带有精确时间戳的SRT文件。

## 环境准备

1. **安装Python**：确保已安装Python 3.7或更高版本。
2. **安装依赖库**：
   ```bash
   pip install pydub speech_recognition transformers librosa ffmpeg-python

使用方法
步骤1：准备输入文件
准备好输入的音频文件（例如input_file.m4a）。

步骤2：运行脚本
运行主脚本 main.py：

bash
python main.py input_file.m4a
脚本将执行以下操作：

分割音频文件。
转录音频片段。
生成SRT文件。

步骤3：查看输出文件
程序将创建一个与输入文件同名目录_seg，并生成srt和txt文件。