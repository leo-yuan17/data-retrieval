import gradio as gr

# 定义处理聊天的函数
def chat_function(message, history):
    if not message.strip():
        return history  # 忽略空消息
    response = f"你说：{message}"
    # 确保历史记录是一个列表，每个元素是一个元组 [用户消息, 系统回复]
    history = history or []  # 如果历史记录为空，初始化为空列表
    history.append([message, response])  # 添加新消息和回复
    return history

# 定义处理文件上传的函数
def file_upload_function(file):
    if not file:
        return "未上传文件"
    file_info = f"文件名: {file.name}\n大小: {round(file.size / 1024, 2)} KB"
    return file_info

# 创建聊天和文件上传的界面
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("### 文件上传与聊天")
    with gr.Row():
        # 左侧聊天功能
        with gr.Column(scale=2):
            gr.Markdown("#### 聊天")
            chat_interface = gr.Chatbot(label="聊天")
            msg_input = gr.Textbox(label="输入消息", placeholder="请输入消息...")
            chat_btn = gr.Button("发送")
            chat_btn.click(
                chat_function,
                inputs=[msg_input, chat_interface],
                outputs=[chat_interface]
            )
        # 右侧文件上传功能
        with gr.Column(scale=1):
            gr.Markdown("#### 文件上传")
            file_input = gr.File(label="上传文件")
            file_output = gr.Textbox(label="文件信息")
            file_input.change(
                file_upload_function,
                inputs=[file_input],
                outputs=[file_output]
            )

# 启动应用
demo.launch()
