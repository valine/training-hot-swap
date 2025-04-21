# Training Hot Swap

This is an example of how to hotswap PyTorch training code without unloading your model weights from VRAM.

For large LLMs it can take upwards of 30 seconds to load a model from disk to VRAM. Waiting 30 seconds every time you want to rerun your script slows down development. This is a barebones implementation of a method to keep large models in VRAM even after your training script exits. If a model reload is necessary, it happens in the background after exit ensuring the model will be ready immediately the next time your script is run.

This works by spawning a second process that stays active after your target script exits. The script you change is not run directly. Instead, this background process runs the code on your behalf using Python's eval().

This can also be used over a VPN for remote code execution. IntelliJ's remote SSH interpreter is quite buggy and not ideal for seamless remote development. Configure model_server.py to run on a remote machine, and run client.py on your development machine. Debugging with the IntelliJ debugger is supported in this configuration as well, enabling an almost seamless development experience with scripts that run instantly and are easily debuggable.

---
## GUI example

Some work has also been done to ensure compatibility with the DearImgui Python bindings. UI code can be submitted to the server along with your training script. I personally like to build out UI for my training scripts to monitor progress, loss over time, and enable easy evaluation. Submitting your UI code along with your training code ensures that your app will launch instantly.

Here's a GUI from an app that displays intermediate output of Mistral 7B. It takes about 0.32 seconds on my machine from when I run the code to when I can interact with the model, and that's including initializtion time for the GUI.
![Screenshot from 2024-12-06 01-35-09](https://github.com/user-attachments/assets/fe38bcb0-0a37-4731-a565-9a785f0885b0)
As an aside, you can find more transformer visualizations of mine here: https://x.com/lukasvaline

---
## Usage
Set your model download location in model_server.py.

Compatible with IntelliJ debug server. Set your debug server port to 5678.

To begin using this in your development simply swap your .from_pretrained call and reference the global variable 'model'

This code goes away:
```python
model = MistralForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype=torch.float16,
    device_map=device,
    use_flash_attention_2=False,
    config=self.config,
)
```

And is replaced with:
```python
def get_model(self):
    """Get model either from global context"""
    global model  # Reference the global model variable

    try:
        # Check if model exists in global scope
        model
    except NameError:
        return None

    return model

model = get_model()
```

How to run:
Launch the server and keep it running
```bash
training-hot-swap$ python model_server.py 
```
Submit the training code to the server
```bash
training-hot-swap$ python client.py ./src ./src/sample_train.py
```
---
## Other considerations 

This script is a major potential security vulnerability. This is a server which by design executes arbitrary code. Don't expose this server to the internet directly. 
