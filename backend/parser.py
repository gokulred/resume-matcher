import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
import nest_asyncio

load_dotenv()
nest_asyncio.apply()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")



def parse_resume(file_path: str)->str:

    if not api_key:
        raise ValueError("Key not foud")

    try:
        parser = LlamaParse(result_type="markdown",
        verbose=True
        )
        documents = parser.load_data(file_path)
        full_text = "\n".join([doc.text for doc in documents])
        return full_text
    except Exception as e:
      print (f"Error parsing the pdf: {str(e)}") 
      return None
    

     
      



    
