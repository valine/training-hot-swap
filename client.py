import atexit
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import psutil
import pydevd_pycharm

class ServerManager:
    def __init__(self, port=12345):
        self.port = port
        self.server_process = None
        self.server_script = None
        atexit.register(self.cleanup)

    def find_server_script(self):
        """Find the server script relative to this client script"""
        current_dir = Path(__file__).parent
        server_script = current_dir / 'server.py'
        if server_script.exists():
            return str(server_script)
        raise FileNotFoundError("server.py not found in the same directory as client.py")

    def is_port_in_use(self):
        """Check if the server port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.port)) == 0

    def wait_for_server(self, timeout=30):
        """Wait for server to start accepting connections"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_in_use():
                return True
            time.sleep(0.1)
        return False

    def ensure_server_running(self):
        """Start the server if it's not already running"""
        if self.is_port_in_use():
            print("Server already running")
            return

        try:
            self.server_script = self.find_server_script()
            print(f"Starting server from {self.server_script}...")

            # Start server process
            self.server_process = subprocess.Popen(
                [sys.executable, self.server_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start
            if not self.wait_for_server():
                raise TimeoutError("Server failed to start within timeout period")

            print("Server started successfully")

        except Exception as e:
            print(f"Failed to start server: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up server process on exit"""
        if self.server_process:
            try:
                # Try graceful shutdown first
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.server_process.kill()
                    self.server_process.wait()

                # Clean up any zombie processes
                for child in psutil.Process(self.server_process.pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

            except psutil.NoSuchProcess:
                pass  # Process already terminated

            print("Server shutdown complete")
            self.server_process = None


class DebugClient:
    def __init__(self, port=12345, debug_port=5678):
        self.port = port
        self.debug_port = debug_port
        self.server_manager = ServerManager(port)

    def is_debugging(self) -> bool:
        """Returns True if the script is being run in debug mode."""
        # Check for debugger trace function
        if hasattr(sys, 'gettrace') and sys.gettrace():
            return True

        # Check for common debugger environment variables
        debug_env_vars = [
            'PYDEVD_LOAD_VALUES_ASYNC',  # PyCharm/IntelliJ
            'PYTHONBREAKPOINT',  # VSCode and others
            'REMOTE_DEBUG',  # Remote debugging
            'PYCHARM_DEBUG',  # PyCharm specific
            'DEBUGPY_PROCESS_GROUP',  # VSCode debugpy
        ]

        return any(var in os.environ for var in debug_env_vars)

    def collect_project_files(self, project_root: str) -> Dict[str, str]:
        """Collect all Python files in the project directory."""
        project_files = {}
        for root, _, files in os.walk(project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        project_files[file_path] = f.read()
        return project_files

    def send_project_to_server(self, project_root: str, main_file: str):
        """Send entire project to the server and receive streamed output."""
        self.server_manager.ensure_server_running()
        project_files = self.collect_project_files(project_root)

        payload = {
            'project_files': project_files,
            'main_file': main_file,
            'project_root': project_root,
            'debug': self.is_debugging(),
        }

        print(f"Sending project to server at port {self.port}...")

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(1.0)  # Add timeout for recv operations
        client_socket.connect(('localhost', self.port))

        try:
            data = pickle.dumps(payload)
            size = len(data)
            client_socket.send(size.to_bytes(8, byteorder='big'))
            client_socket.send(data)

            while True:
                try:
                    size_data = client_socket.recv(8)
                    if not size_data:
                        break

                    size = int.from_bytes(size_data, byteorder='big')
                    data = b""
                    while len(data) < size:
                        try:
                            chunk = client_socket.recv(min(size - len(data), 4096))
                            if not chunk:
                                raise ConnectionError("Connection closed by server")
                            data += chunk
                        except socket.timeout:
                            continue
                        except (ConnectionError, BrokenPipeError):
                            print("\nThe server has disconnected")
                            sys.exit(1)

                            return

                    if data:
                        try:
                            message_type, content = pickle.loads(data)
                            if message_type == 'stdout':
                                print(content, end='')
                            elif message_type == 'error':
                                print(f"Error: {content}", file=sys.stderr)
                                break
                            else:
                                print("\nServer response:", content)
                                break
                        except pickle.UnpicklingError:
                            print("\nReceived corrupt data from server")
                            break

                except socket.timeout:
                    continue
                except (ConnectionError, BrokenPipeError):
                    print("\nThe server has disconnected")
                    sys.exit(1)
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            client_socket.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <project_root_directory> <main_file>")
        sys.exit(1)

    project_root = os.path.abspath(sys.argv[1])
    main_file = os.path.abspath(sys.argv[2])

    if not os.path.isdir(project_root):
        print(f"Error: {project_root} is not a directory")
        sys.exit(1)
    if not os.path.isfile(main_file):
        print(f"Error: {main_file} is not a file")
        sys.exit(1)

    client = DebugClient()
    try:
        client.send_project_to_server(project_root, main_file)
    except KeyboardInterrupt:
        print("\nInterrupted by user")