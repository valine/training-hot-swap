# Training Hot Swap

This is an example of how to hotswap pytorch training code without unloading the model weights from VRAM. 

For large LLMs it can take upwards of 30 seconds to load a model from disk to VRAM. Waiting 30 seconds every time you want to rerun your script slows down development. This is a barebones implementation of a method to keep large models in VRAM even after your traing script exits. If a model reload is nessesary it happens in the background after exit ensuring the model will be reasy immediately the next time your script is run. 

This works by spawing a second process that stays active after your target script exists. The sceipt you change is not run directly, rather this background process runs the code on your behalf using pythons eval().

Set your model download location in model_server.py.

Compatible with Intellj debug server. Set your debug server port to 5678

How to run:
Launch the server and keep it running
```bash
training-hot-swap$ python model_server.py 
```
Submit the training code to the server
```bash
training-hot-swap$ python client.py ./src ./src/sample_train.py
```
