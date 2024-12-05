# Training Hot Swap

This is an example of how to hotswap pytorch training code without unloading the model weights from VRAM. 

How to run:
Launch the server and keep it running
```bash
training-hot-swap$ python server.py 
```
Submit the training code to the server
```bash
training-hot-swap$ python client.py ./src ./src/sample_train.py
```