import chromadb
import logging
class data_vector:
    def __init__(self,path:str,client_name:str,collection_name:str,**kwargs):
        self.path = path
        self.kwargs = kwargs
        self.client_name = client_name
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path,name=client_name,**self.kwargs)
        self.collection = None
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logging.info(f"Collection {self.collection_name} loaded\n")
        except :
            logging.warning(f"Collection {self.collection_name} not exist \n")
            try:
                self.collection = self.client.create_collection(self.collection_name)
                logging.info(f"Collection {self.collection_name} created\n")
            except:
                logging.error(f"Error creating collection {self.collection_name}\n")
                
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('chromadb').setLevel(logging.CRITICAL)
    dv = data_vector("chromadb","test")


            
        