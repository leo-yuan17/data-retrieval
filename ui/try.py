import gradio as gr
from gradio_pdf import PDF

def qa(question: str, doc: str) -> str:
    # 在这里实现处理逻辑，例如解析 PDF 文档并回答问题
    # 这里只是返回一个示例答案
    return f"Question: {question}\nDocument: {doc}\nAnswer: This is a placeholder answer."

# 创建 Gradio 界面
demo = gr.Interface(
    fn=qa,  # 处理逻辑函数
    inputs=[gr.Textbox(label="Question"), PDF(label="Document")],  # 输入组件
    outputs=gr.Textbox(label="Answer")  # 输出组件
)

if __name__ == "__main__":
    demo.launch()
