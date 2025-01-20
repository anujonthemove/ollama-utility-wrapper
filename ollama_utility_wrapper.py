import psutil
import shutil
from tqdm import tqdm
from ollama import ps, list, pull, chat
from ollama import ListResponse, ProcessResponse

class OllamaUtilityWrapper:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._ollama_installed = None
            self._ollama_running = None
            self._initialized = True

    def is_ollama_installed(self) -> bool:
        if self._ollama_installed is None:
            self._ollama_installed = shutil.which("ollama") is not None
        return self._ollama_installed

    def is_ollama_running(self) -> bool:
        if self._ollama_running is None:
            self._ollama_running = any(
                'ollama' in process.info['name'].lower()
                for process in psutil.process_iter(['pid', 'name'])
            )
        return self._ollama_running

    def list_all_downloaded_models(self) -> None:
        response: ListResponse = list()
        for model in response.models:
            print(f'Name: {model.model}')
            print(f'Size (MB): {(model.size.real / 1024 / 1024):.2f}')
            if model.details:
                print(f'Format: {model.details.format}')
                print(f'Family: {model.details.family}')
                print(f'Parameter Size: {model.details.parameter_size}')
                print(f'Quantization Level: {model.details.quantization_level}')
                print('\n')

    def download_model(self, model_name: str) -> None:
        current_digest = '' 
        progress_bars = {}

        for progress in pull(model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in progress_bars:
                progress_bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in progress_bars and (total := progress.get('total')):
                progress_bars[digest] = tqdm(total=total, desc=f'Downloading {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                progress_bars[digest].update(completed - progress_bars[digest].n)

            current_digest = digest

        for bar in progress_bars.values():
            bar.close()

    def load_model(self, model_name: str) -> bool:
        try:
            print("Loading model... \n")
            chat_response = chat(model=model_name, messages=[{'role': 'user', 'content': 'Which model is this?'}])
            return True
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return False
        
    def get_loaded_model_info(self) -> None:
        response: ProcessResponse = ps()
        if not response.models:
            print('No models are currently loaded.')
            return 
        for model in response.models:
            print(f'Model: {model.model}')
            print(f'Digest: {model.digest}')
            print(f'Expires at: {model.expires_at}')
            print(f'Size: {model.size}')
            print(f'Size VRAM: {model.size_vram}')
            print(f'Details: {model.details}')
            print('\n')

    def is_model_downloaded(self, model_name: str) -> bool:
        response: ListResponse = list()
        return any(model_name == model.model for model in response.models)

    def initialize_ollama_model(self, model_name: str) -> None:
        if not self.is_ollama_installed():
            print("Ollama is not installed. Please install Ollama and try again.")
            return

        if not self.is_ollama_running():
            print("Ollama is not running. Please start the Ollama service and try again.")
            return
        
        try:
            if self.is_model_downloaded(model_name=model_name):
                print("Model is already available on your machine.")
            else:
                print(f"Model does not exist, attempting to download: {model_name}")
                self.download_model(model_name)

            if self.load_model(model_name):
                print("Model loaded successfully. \n")
                self.get_loaded_model_info()
            else:
                print(f"Model {model_name} could not be loaded.")
        except Exception as e:
            print(f"An error occurred: {e}")