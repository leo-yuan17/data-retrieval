import gradio as gr

# 初始化一个全局列表来保存文件名
uploaded_files = []

# 定义处理函数
def display_filenames(files):
    global uploaded_files
    
    # 获取上传的所有文件名并更新历史文件列表
    filenames = [file.name for file in files]
    uploaded_files.extend(filenames)  # 将新上传的文件名添加到历史文件列表中
    
    # 返回所有文件名（包括历史上传的）
    return "\n".join(uploaded_files)

# 创建 Gradio 界面
demo = gr.Interface(
    fn=display_filenames,  # 处理函数
    inputs=gr.Files(label="上传文件", file_count="multiple"),  # 允许上传多个文件
    outputs=gr.Textbox(label="文件名列表",max_length=25,lines = 25),  # 输出文件名
)

if __name__ == "__main__":
    demo.launch()
