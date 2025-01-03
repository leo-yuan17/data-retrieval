import chromadb
import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import logging
import json


class data_loader:

    def __init__(self, path: str, client_name: str, collection_name: str,
                 **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.client_name = client_name
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path,
                                                **self.kwargs)
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
        
    def load_from_json(self,json_path:str):
        """
        Load the content from the json file
        """
        self.json_path = json_path
        with open(self.json_path, 'r',encoding="utf-8") as f:
            self.content = json.load(f)
        return self.content
    
    def add_json_in_db(self):
        """
        turn the json content to a database
        """
            
            
if __name__ == "__main__":
    data_loader = data_loader(path="data", client_name="chroma_db",
                              collection_name="pdf_data")
    # data_loader.extract_text_from_pdf("data/1-s2.0-S0169500222006870-main.pdf")
    content = data_loader.load_from_json(r"output\data\1-s2_content_list.json")
    print(content)
    
            