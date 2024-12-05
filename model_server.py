import copy
import importlib.util
# intellij_debug_startup.py
import os
import pickle
import queue
import shutil
import signal
import socket
import sys
import threading
import traceback
import types
import warnings
from contextlib import contextmanager
from datetime import datetime
from queue import Queue
from typing import Dict

import pydevd_pycharm
import torch
from transformers import AutoTokenizer, MistralForCausalLM, AutoConfig
from transformers import logging as hf_logging

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

class ExecutionTask:
    def __init__(self, project_files, main_file, project_root, debug, client_socket):
        self.project_files = project_files
        self.main_file = main_file
        self.project_root = project_root
        self.debug = debug
        self.client_socket = client_socket
        self.message_queue = Queue()
        self.client_handler = ClientHandler(client_socket, self.message_queue)


class ServerListener(threading.Thread):
    def __init__(self, host='localhost', port=12345, model_server=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.running = False
        self.task_queue = Queue()
        self.server_socket = None
        self.model_server = model_server

    def run(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Server listening on port {self.port}")

            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    try:
                        client_socket, _ = self.server_socket.accept()
                    except socket.timeout:
                        continue

                    size_data = client_socket.recv(8)
                    size = int.from_bytes(size_data, byteorder='big')

                    data = b""
                    while len(data) < size:
                        chunk = client_socket.recv(min(size - len(data), 4096))
                        if not chunk:
                            break
                        data += chunk

                    if data:
                        payload = pickle.loads(data)
                        task = ExecutionTask(
                            payload['project_files'],
                            payload['main_file'],
                            payload['project_root'],
                            payload['debug'],
                            client_socket
                        )

                        if self.model_server and self.model_server.current_task:
                            os.kill(os.getpid(), signal.SIGINT)

                        self.task_queue.put(task)

                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")

        finally:
            if self.server_socket:
                self.server_socket.close()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()


class StreamingStdout:
    def __init__(self, message_queue):
        self.message_queue = message_queue
        self.original_stdout = sys.stdout
        self.closed = False

    def write(self, text):
        if self.closed:
            self.original_stdout.write(text)
            return

        try:
            self.original_stdout.write(text)
            self.message_queue.put(('stdout', text))
        except:
            self.closed = True
            self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

    def close(self):
        self.closed = True

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.close()


class ClientHandler(threading.Thread):
    def __init__(self, client_socket, message_queue):
        super().__init__(daemon=True)
        self.client_socket = client_socket
        self.message_queue = message_queue
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            while self.running:
                try:
                    message_type, content = self.message_queue.get(timeout=0.1)
                    if not self.client_socket._closed:
                        message = pickle.dumps((message_type, content))
                        size = len(message)
                        self.client_socket.send(size.to_bytes(8, byteorder='big'))
                        self.client_socket.send(message)
                except queue.Empty:
                    continue
                except (socket.error, BrokenPipeError):
                    break
        finally:
            if not self.client_socket._closed:
                self.client_socket.close()

class DebugSymbolManager:
    def __init__(self):
        self.debug_port = 5678
        self.debugger_attached = False

    def attach_debugger(self):
        """Attach to PyCharm debugger"""
        try:
            pydevd_pycharm.settrace('localhost',
                                    port=self.debug_port,
                                    stdoutToServer=True,
                                    stderrToServer=True,
                                    suspend=False,
                                    trace_only_current_thread=False)
            self.debugger_attached = True
            print(f"Debugger attached on port {self.debug_port}")
        except Exception as e:
            pass

    def detach_debugger(self):
        """Detach the PyCharm debugger"""
        if self.debugger_attached:
            try:
                pydevd_pycharm.stoptrace()
                self.debugger_attached = False
                print("Debugger detached")
            except Exception as e:
                print(f"Failed to detach debugger: {e}")


class ServerShutdown(Exception):
    """Custom exception for handling graceful server shutdown"""
    pass


class ModelServer:
    def __init__(self, model_name=MODEL_NAME, tokenizer_path=MODEL_NAME,
                 port=12345):
        self.port = port
        self.running = False
        self.debug_manager = DebugSymbolManager()
        self.current_task = None
        self.listener = ServerListener(port=port, model_server=self)

        warnings.filterwarnings("ignore", category=UserWarning, module='transformers.*')
        hf_logging.set_verbosity_error()

        self.model = None
        self.model_ref = None
        self.load_model(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = "</s>"

        self._original_class = self.model.__class__
        print("Model and tokenizer loaded and ready")

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutdown signal received. Cleaning up...")
        self.running = False
        self.listener.stop()
        if self.current_task:
            try:
                self.current_task.client_handler.stop()
                if not self.current_task.client_socket._closed:
                    self.current_task.client_socket.close()
            except:
                pass
        if not self.listener.task_queue.empty():
            return
        raise ServerShutdown()

    @contextmanager
    def client_connection(self, client_socket):
        """Context manager for handling client connections"""
        stdout_handler = None
        try:
            with StreamingStdout(client_socket) as stdout_handler:
                yield client_socket
        except (ConnectionError, BrokenPipeError) as e:
            print(f"Client disconnected: {e}")
            if stdout_handler:
                stdout_handler.close()
        except Exception as e:
            print(f"Error handling client connection: {e}")
            if not client_socket._closed:
                try:
                    error_msg = pickle.dumps(('error', str(e)))
                    size = len(error_msg)
                    client_socket.send(size.to_bytes(8, byteorder='big'))
                    client_socket.send(error_msg)
                except:
                    pass
        finally:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            client_socket.close()

    def load_model(self, model_name):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Less than 2 GPUs available")

        devices_in_order = ["cpu", "cuda:2"]
        config = AutoConfig.from_pretrained(model_name)
        models = {}

        for idx, model_key in enumerate(['model']):
            device = devices_in_order[idx]
            model = MistralForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                use_flash_attention_2=False,
                config=config,
            )
            models[model_key] = model

        self.model = models['model']
        model_copy = copy.deepcopy(self.model)

        self.model = self.model.to('cuda:0')
        self.model_ref = model_copy.to('cuda:2')

    def generate_test_text(self, model=None, tokenizer=None):
        prompt = "Write a story about a dragon that terrorizes a village."

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False).to(
            model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        num_tokens_to_generate = 3
        # Generate text
        print("\nGenerating response...")

        outputs = model.generate(
            input_ids,
            max_length=num_tokens_to_generate + input_ids.shape[-1],
            temperature=1.0,
            repetition_penalty=1.1,
            do_sample=False,
            attention_mask=attention_mask,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode and print the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{generated_text}")

    def create_module(self, module_code: str, module_name: str) -> types.ModuleType:
        """Create a new module with the given code."""
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)

        # Create a code object with filename information
        code_obj = compile(module_code, f"/home/lukas/Desktop/latent-descent/src/lsd/{module_name.split('.')[-1]}.py",
                           'exec')

        module.__dict__['model'] = self.model
        module.__dict__['tokenizer'] = self.tokenizer
        module.__dict__['model_ref'] = self.model_ref

        # Execute the compiled code object instead of the string
        exec(code_obj, module.__dict__)
        return module

    def execute_project(self, project_files: Dict[str, str], main_file: str, project_root: str, debug: bool):
        start_time = datetime.now()
        debug_root = None
        try:

            sys.path.insert(0, project_root)
            modules = {}

            for module_name in list(sys.modules.keys()):
                if module_name not in sys.builtin_module_names and \
                        hasattr(sys.modules[module_name], '__file__') and \
                        sys.modules[module_name].__file__ and \
                        project_root in str(sys.modules[module_name].__file__):
                    del sys.modules[module_name]

            for file_path, content in project_files.items():
                if not file_path.endswith('.py'):
                    continue

                rel_path = os.path.relpath(file_path, project_root)
                module_name = os.path.splitext(rel_path)[0].replace(os.sep, '.')

                module = self.create_module(content, module_name)
                modules[module_name] = module
                sys.modules[module_name] = module

            main_module_path = os.path.relpath(main_file, project_root)
            main_module_name = os.path.splitext(main_module_path)[0].replace(os.sep, '.')

            if debug:
                # Add all potentially relevant paths without overwriting project_root
                paths_to_add = [
                    project_root,  # Root directory
                ]

                # Print for debugging
                print("Project root:", project_root)
                print("Paths being added:", paths_to_add)

                sys.path.extend([p for p in paths_to_add if p not in sys.path])

                # debug_root = self.debug_manager.setup_debugging(project_files, project_root, main_file)
                self.debug_manager.attach_debugger()

            if main_module_name in modules:
                main_module = modules[main_module_name]
                if hasattr(main_module, 'main'):
                    main_module.main()
                else:
                    print("Warning: No main() function found in main module")
            else:
                raise ValueError(f"Main module {main_module_name} not found in project files")

            torch.cuda.empty_cache()
            # Unregister signal handlers
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            width = shutil.get_terminal_size().columns
            run_time = datetime.now() - start_time
            # Print current time and vram usage
            # AM PM time
            print("Completed at:", datetime.now().strftime("%Y-%m-%d %I:%M %p"))
            # Total run time
            print("Total run time:", str(run_time).split(".")[0])

            divider = '▰' * width
            print(divider)

            return "Project execution completed successfully"

        except Exception as e:
            print(f"Error during execution: {e}")
            exception = sys.exc_info()
            stack_trace = traceback.format_exc()
            # You can log the error here if needed
            print("Exception in thread:", stack_trace)
            raise

        finally:
            if debug:
                self.debug_manager.detach_debugger()  # Add this line
            sys.path.remove(project_root)
            for module_name in list(sys.modules.keys()):
                if module_name in modules:
                    del sys.modules[module_name]


    @contextmanager
    def server_socket_context(self):
        """Context manager for server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Set SO_REUSEADDR option to allow quick server restart
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('localhost', self.port))
            self.server_socket.listen(1)
            yield self.server_socket
        finally:
            print("Closing server socket...")
            self.server_socket.close()
            self.server_socket = None

    def start(self):
        self.running = True
        self.listener.start()

        try:
            while self.running:
                try:
                    try:
                        self.current_task = self.listener.task_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    # Start the client handler thread for this task
                    self.current_task.client_handler.start()

                    try:
                        # Use StreamingStdout with the task's message queue
                        with StreamingStdout(self.current_task.message_queue):
                            result = self.execute_project(
                                self.current_task.project_files,
                                self.current_task.main_file,
                                self.current_task.project_root,
                                self.current_task.debug
                            )
                        # Send final result
                        self.current_task.message_queue.put(('result', result))
                    except Exception as e:
                        print(f"Task execution error: {e}")
                        self.current_task.message_queue.put(('error', str(e)))
                    finally:
                        # Stop the client handler and clean up
                        self.current_task.client_handler.stop()
                        self.current_task = None

                except Exception as e:
                    print(f"Main loop error: {e}")

        except ServerShutdown:
            print("Server shutdown completed")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
        finally:
            self.running = False
            self.listener.stop()
            if self.current_task:
                self.current_task.client_handler.stop()
                if not self.current_task.client_socket._closed:
                    self.current_task.client_socket.close()

if __name__ == "__main__":
    server = ModelServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutdown requested... closing server")
