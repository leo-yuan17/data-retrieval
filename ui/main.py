import gradio as gr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.embedding_2 import load_embeddingmodel
from dotenv import load_dotenv
from utils.datavector import *

# 初始化一个全局列表来保存文件名
uploaded_files = []
load_dotenv("api.env")
llm = llms.load_llm(model="deepseek-chat",
                     api_key=os.environ.get("DEEPSEEK_API"),
                     url="https://api.deepseek.com")
embed = load_embeddingmodel().get_embedding_model()
tokenizer = load_embeddingmodel().get_tokenizer()
data_loader = data(path="data",
                   client_name="chroma_db",
                   collection_name="pdf_data")


def handle_messages(messages, history, patient_condition, tokens=4096):
    if messages["text"]:
        response = data_loader.query(messages["text"],patient_condition,
                                    embed,
                                    llm=llm,
                                    tokenizer=tokenizer)
    elif messages["files"]:
        for file in messages["files"]:
            data_loader.extract_text_from_pdf(file,embed,tokenizer)

        response = "文件已上传"
    return response


textbox = gr.MultimodalTextbox(file_count="multiple",
                               placeholder="上传文件或输入文本",
                               render=True)
with gr.Blocks() as demo:
    patient_condition = gr.Textbox(" ", label="病人情况")

    gr.ChatInterface(
        handle_messages,
        additional_inputs=[patient_condition],
        type="messages",
        textbox=textbox,
        save_history=True,
        flagging_mode="manual",
        flagging_options=["准确", "不准确"],
    )
    textbox
if __name__ == "__main__":
    demo.launch()
