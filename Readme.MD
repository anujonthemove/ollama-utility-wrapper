# Ollama Utility Wrapper

The **Ollama Utility Wrapper** is a Python class designed to simplify the management of language models using the [ollama-python](https://github.com/ollama/ollama-python) library. This utility consolidates multiple functionalities such as verifying installation, downloading models, listing loaded models, and more, all within a single reusable interface.

## Table of Contents
- [Why This Exists](#why-this-exists)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Example Workflow](#example-workflow)
- [Contributing](#contributing)
- [License](#license)



## Why This Exists

Managing models with Ollama often requires switching between terminal commands like `run`, `stop`, `pull`, `list`, and `ps`. While these commands are effective, they can be cumbersome when integrating Ollama into Python-based workflows.

The **Ollama Utility Wrapper** was created to:
- **Streamline model management**: Provide a Python-first interface to manage models without relying on terminal commands.
- **Enhance productivity**: Reduce the friction of switching between terminal and Python code.
- **Simplify workflows**: Encapsulate common tasks such as checking installation, downloading models, and listing loaded models.



## Features

This utility provides the following functionalities:
1. **Check Installation and Service Status**:
   - Verify if Ollama is installed.
   - Check if the Ollama service is running.

2. **List Downloaded Models**:
   - View details of all models already available on your system.

3. **Download Models with Progress**:
   - Download models with a real-time progress bar for better transparency.

4. **Load Models**:
   - Load a downloaded model into memory for use in applications.

5. **Display Loaded Models**:
   - Fetch and display details of all currently loaded models.

6. **One-Step Initialization**:
   - Automate the process of verifying installation, downloading missing models, and loading them into memory.



## Installation

1. Create a virtual environment (optional but recommended):
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
2. Install the dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Ensure Ollama is installed and running on your machine:
    - [Ollama Installation Guide](https://ollama.com/)


## Usage

Here’s how you can use the utility:

1. Import the utility class:
    
    ```python
    
    from ollama_utility_wrapper import OllamaUtilityWrapper
    
    ```
    
2. Instantiate the wrapper:
    
    ```python
    wrapper = OllamaUtilityWrapper()
    
    ```
    
3. Perform operations:
    - Check if Ollama is installed and running:
        
        ```python
        if wrapper.is_ollama_installed() and wrapper.is_ollama_running():
            print("Ollama is ready!")
        else:
            print("Please install or start Ollama.")
        
        ```
        
    - List downloaded models:
        
        ```python
        wrapper.list_all_downloaded_models()
        
        ```
        
    - Download and load a model:
        
        ```python
        wrapper.initialize_ollama_model("example-model")
        
        ```
        



## API Reference

### `OllamaUtilityWrapper`

The main class encapsulating all utility functions.

### Methods:

1. **`is_ollama_installed()`**
    - Checks if Ollama is installed on the system.
    - **Returns**: `True` if installed, `False` otherwise.
2. **`is_ollama_running()`**
    - Checks if the Ollama service is running.
    - **Returns**: `True` if running, `False` otherwise.
3. **`list_all_downloaded_models()`**
    - Lists all models downloaded on the system with details like size and format.
4. **`download_model(model_name: str)`**
    - Downloads the specified model and shows a progress bar.
5. **`load_model(model_name: str)`**
    - Loads a downloaded model into memory.
6. **`get_loaded_model_info()`**
    - Displays details of currently loaded models.
7. **`is_model_downloaded(model_name: str)`**
    - Checks if a specific model is already downloaded.
8. **`initialize_ollama_model(model_name: str)`**
    - Combines installation check, model download, and loading into one step.



## Example Workflow

Here’s a complete example to demonstrate the utility in action:

```python
from ollama_utility_wrapper import OllamaUtilityWrapper

# Initialize the wrapper
wrapper = OllamaUtilityWrapper()

# Step 1: Check if Ollama is ready
if not wrapper.is_ollama_installed():
    print("Ollama is not installed. Please install it first.")
elif not wrapper.is_ollama_running():
    print("Ollama is not running. Start the service and try again.")
else:
    print("Ollama is ready!")

# Step 2: List available models
print("\nListing all downloaded models:")
wrapper.list_all_downloaded_models()

# Step 3: Initialize a model
model_name = "example-model"
print(f"\nInitializing model: {model_name}")
wrapper.initialize_ollama_model(model_name)

```


## Contributing

Contributions are welcome!

## Issues

If you encounter any issues while using the utility, please [open an issue](https://github.com/anujonthemove/ollama-utility-wrapper/issues) in the repository. Include details about the problem and any error messages you encountered.