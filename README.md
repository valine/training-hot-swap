# Training Hot Swap

This is an example of how to hotswap PyTorch training code without unloading the model weights from VRAM.

For large LLMs it can take upwards of 30 seconds to load a model from disk to VRAM. Waiting 30 seconds every time you want to rerun your script slows down development. This is a barebones implementation of a method to keep large models in VRAM even after your training script exits. If a model reload is necessary, it happens in the background after exit ensuring the model will be ready immediately the next time your script is run.

This works by spawning a second process that stays active after your target script exits. The script you change is not run directly. Instead, this background process runs the code on your behalf using Python's eval().

This can also be used over a VPN for remote code execution. IntelliJ's remote SSH interpreter is quite buggy and not ideal for seamless remote development. Configure model_server.py to run on a remote machine, and run client.py on your development machine. Debugging with the IntelliJ debugger is supported in this configuration as well, enabling an almost seamless development experience with scripts that run instantly and are easily debuggable.

Some work has also been done to ensure compatibility with the DearImgui Python bindings. UI code can be submitted to the server along with your training script. I personally like to build out UI for my training scripts to monitor progress, loss over time, and enable easy evaluation. Submitting your UI code along with your training code ensures that your app will launch instantly.

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
    """Get model either from global context or load it fresh"""
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
