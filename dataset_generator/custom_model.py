from typing import List
import aiohttp
import requests
import openai
from langchain_core.embeddings import Embeddings

# langchain_core.embeddings.Embeddings: https://python.langchain.com/docs/how_to/custom_embeddings/
# openai completions usage:             https://platform.openai.com/docs/guides/completions

class CustomEmbeddingModelWithAPI(Embeddings):
    """ Custom Embedding Model with OpenAI API """
    api_base:      str
    model_name:    str
    embedding_dim: int
    
    def __init__(self, api_base: str, model_name: str, embedding_dim: int = None):
        
        if not isinstance(api_base, str):
            raise TypeError("api_base is not a string")
        if not isinstance(model_name, str):
            raise TypeError("model_name is not a string")
        self.api_base = api_base if api_base[-1] != '/' else api_base[:-1]
        self.model_name = model_name
        
        if embedding_dim is None:
            # embed test_text and get the output embedding dimension
            test_text = "hello"
            output_vect = self.embed_query(test_text)
            self.embedding_dim = len(output_vect)
        else:
            self.embedding_dim = embedding_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """ Embed search docs """
        url = f"{self.api_base}/embeddings"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"{self.model_name}",
            "input": texts
        }
        response = requests.post(url, json=data, headers=headers)
        return [result['embedding'] for result in response.json()['data']]

    def embed_query(self, text: str) -> List[float]:
        """ Embed query text """

        url = f"{self.api_base}/embeddings"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"{self.model_name}",
            "input": f"{text}"
        }

        response = requests.post(url, json=data, headers=headers)
        return response.json()['data'][0]['embedding']

    # optional: add custom async implementations here
    """ you can also delete these, and the base class will
    use the default implementation, which calls the sync
    version in an async executor: """

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """ Asynchronous Embed search docs """
        url = f"{self.api_base}/embeddings"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"{self.model_name}",
            "input": texts
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                response_data = await response.json()
                return [result['embedding'] for result in response_data['data']]

    async def aembed_query(self, text: str) -> List[float]:
        """ Asynchronous Embed query text """
        url = f"{self.api_base}/embeddings"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"{self.model_name}",
            "input": text
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                response_data = await response.json()
                return response_data['data'][0]['embedding']

class CustomChatModel:
    """ Custom Chat Model with OpenAI API """
    base_url:      str
    api_key:       str
    model_name:    str
    client:        openai.OpenAI
    
    def __init__(self, model_name: str, base_url: str, api_key: str = 'no-need'):
        if not isinstance(base_url, str):
            raise TypeError("base_url is not a string")
        if not isinstance(api_key, str):
            raise TypeError("api_key is not a string")
        if not isinstance(model_name, str):
            raise TypeError("model_name is not a string")
        
        self.base_url = base_url if base_url[-1] == '/' else (base_url + '/')
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key = self.api_key, base_url = self.base_url)
        
    def invoke(self, messages: list[dict[str, str]], temperature: float = 0.7, stream: bool = False):

        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = temperature, 
            stream = stream
        )
        return completion

if __name__ == '__main__':
    pass

