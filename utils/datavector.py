import chromadb
import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import logging
import json
import llms


class data:

    def __init__(self, path: str, client_name: str, collection_name: str,
                 **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.client_name = client_name
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path, **self.kwargs)
        self.collection = None
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logging.info(f"Collection {self.collection_name} loaded\n")
        except:
            logging.warning(f"Collection {self.collection_name} not exist \n")
            try:
                self.collection = self.client.create_collection(
                    self.collection_name)
                logging.info(f"Collection {self.collection_name} created\n")
            except:
                logging.error(
                    f"Error creating collection {self.collection_name}\n")

    def extract_text_from_pdf(self, data_path: str):
        """
        Extract text from pdf file
        save the extracted text in a markdown file
        save the extracted images in a folder
        save the extracted content list in a json file
        """
        if os.path.exists(data_path):
            name_without_suff = data_path.split(".")[0]
        local_image_dir, local_md_dir = "output/images", "output"
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = FileBasedDataWriter(
            local_image_dir), FileBasedDataWriter(local_md_dir)
        image_dir = str(os.path.basename(local_image_dir))

        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(data_path)  # read the pdf content

        # proc
        ## Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)

            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)
            ### dump markdown
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        ### dump content list
        pipe_result.dump_content_list(
            md_writer, f"{name_without_suff}_content_list.json", image_dir)
        logging.info(f"Text extracted from {data_path} successfully\n")

    def load_from_json(self, json_path: str):
        """
        Load the content from the json file
        """
        self.json_path = json_path
        with open(self.json_path, 'r', encoding="utf-8") as f:
            self.content = json.load(f)
        return self.content

    def embed_text(self,
                   text,
                   embedding_model,
                   tokenizer,
                   device: str = "cpu"):
        """
        Embed the text using the embedding model
        """
        embedding_model.to(device)
        self.device = device
        inputs = tokenizer(text,
                           padding=True,
                           truncation=True,
                           max_length=512,
                           return_tensors="pt")
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = embedding_model(**inputs_on_device, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
        embeddings = embeddings / embeddings.norm(dim=1,
                                                  keepdim=True)  # normalize
        return embeddings

    def add_json_in_db(self, json_path: str, embedding_model, tokenizer):
        """
        Add the json file to the database
        """
        self.json_path = json_path
        self.ids = 0
        with open(self.json_path, 'r', encoding="utf-8") as f:
            self.content = json.load(f)
        for item in self.content:
            if "text" in item:
                embedding = self.embed_text(item["text"],
                                            embedding_model=embedding_model,
                                            tokenizer=tokenizer)
                self.collection.add(embeddings=embedding.detach().numpy(),
                                    metadatas=[{
                                        "page_idx": item["page_idx"],
                                        "text": item["text"]
                                    }],
                                    ids=str(self.ids))
            elif "image" in item:
                embedding = self.embed_text(item["img_caption"])
                self.collection.add(embeddings=embedding.detach().numpy(),
                                    metadatas=[{
                                        "page_idx":
                                        item["page_idx"],
                                        "img_caption":
                                        item["img_caption"],
                                        "img_path":
                                        item["img_path"]
                                    }],
                                    ids=str(self.ids))
            elif "table" in item:
                embedding = self.embed_text(item["table_caption"])
                self.collection.add(embeddings=embedding.detach().numpy(),
                                    metadatas=[{
                                        "page_idx":
                                        item["page_idx"],
                                        "table_caption":
                                        item["table_caption"],
                                        "table_path":
                                        item["table_path"]
                                    }],
                                    ids=str(self.ids))
            self.ids += 1
        logging.info(
            f"Json file {json_path} added to the database successfully\n")

    def query(self, query: str, embedding_model, tokenizer, llm, top=5):
        """
        Query the database with a query string
        """
        query_embedding = self.embed_text(query,
                                          embedding_model=embedding_model,
                                          tokenizer=tokenizer)
        results = self.collection.query(query_embedding.detach().numpy(),
                                        n_results=top)
        results = results["metadatas"][0]
        content = ""
        for result in results:
            content += result["text"] + " "
        results = llm.get_response(query, content)
        return results


if __name__ == "__main__":
    from embedding_2 import load_embeddingmodel
    llms = llms.load_llm(model="deepseek-chat",
                         api_key="sk-16824413873f4defa607185b05663278",
                         url="https://api.deepseek.com")
    embed = load_embeddingmodel().get_embedding_model()
    tokenizer = load_embeddingmodel().get_tokenizer()
    data_loader = data(path="data",
                       client_name="chroma_db",
                       collection_name="pdf_data")
    # data_loader.extract_text_from_pdf("data/1-s2.0-S0169500222006870-main.pdf")
    content = data_loader.load_from_json(r"output\data\1-s2_content_list.json")
    # data_loader.add_json_in_db(r"output\data\1-s2_content_list.json", embed,
    #    tokenizer)
    results = data_loader.query("what is the main conclusion of this paper",
                                embed,
                                llm=llms,
                                tokenizer=tokenizer)
    print(results)
