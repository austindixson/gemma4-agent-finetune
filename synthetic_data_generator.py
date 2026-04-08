#!/usr/bin/env python3
"""
Synthetic Data Augmentation for Agent Fine-Tuning

This module generates synthetic examples to improve:
1. One-shot coding skill
2. Tool-driven agent loop ability
3. Edge case handling
4. Function calling accuracy
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path

class SyntheticDataGenerator:
    """Generate synthetic training examples for agent fine-tuning"""

    def __init__(self, hermes_dataset_path: str = "./hermes_dataset"):
        self.hermes_path = Path(hermes_dataset_path)
        self.tool_definitions = self._load_tool_definitions()

    def _load_tool_definitions(self) -> List[Dict]:
        """Load tool definitions from Hermes dataset"""
        # Common tools for agent operations
        return [
            {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "search_files",
                "description": "Search for files matching pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def generate_one_shot_coding_example(self, difficulty: str = "medium") -> Dict:
        """
        Generate one-shot coding examples

        Args:
            difficulty: 'easy', 'medium', 'hard'

        Returns:
            Conversation in ShareGPT format
        """
        examples = {
            "easy": [
                {
                    "from": "human",
                    "value": "Write a function to check if a number is even."
                },
                {
                    "from": "gpt",
                    "value": """```python
def is_even(n: int) -> bool:
    '''Check if a number is even.'''
    return n % 2 == 0

# Example usage:
print(is_even(4))  # True
print(is_even(7))  # False
```"""
                }
            ],
            "medium": [
                {
                    "from": "human",
                    "value": "Create a function that reads a JSON file and returns the value for a given key."
                },
                {
                    "from": "gpt",
                    "value": """```python
import json

def get_json_key(filepath: str, key: str):
    '''Read JSON file and return value for key.'''
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get(key)

# Example:
# config.json: {"database_url": "postgres://localhost/mydb", "timeout": 30}
# get_json_key("config.json", "timeout") → 30
```"""
                }
            ],
            "hard": [
                {
                    "from": "human",
                    "value": "Write a function that implements a caching decorator with TTL support and thread-safety."
                },
                {
                    "from": "gpt",
                    "value": """```python
import time
import threading
from functools import wraps
from typing import Dict, Any, Callable

def ttl_cache(ttl: int = 60):
    '''
    Thread-safe caching decorator with TTL support.

    Args:
        ttl: Time to live in seconds
    '''
    cache: Dict[Any, tuple] = {}
    lock = threading.Lock()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))

            with lock:
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < ttl:
                        return result
                    else:
                        del cache[key]

            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, time.time())
            return result

        return wrapper
    return decorator

# Usage:
@ttl_cache(ttl=30)
def expensive_computation(n: int) -> int:
    time.sleep(1)  # Simulate work
    return n * 2
```"""
                }
            ]
        }

        return {
            "conversations": examples[difficulty]
        }

    def generate_codealpaca_style_example(self, category: str = "arrays") -> Dict:
        """
        Generate CodeAlpaca-style coding examples

        Args:
            category: Type of coding task (arrays, strings, algorithms, oop, etc.)

        Returns:
            Conversation in ShareGPT format
        """
        examples = {
            "arrays": [
                {
                    "from": "human",
                    "value": "Create a function that takes an array of integers and returns the sum of all even numbers."
                },
                {
                    "from": "gpt",
                    "value": """```python
def sum_even_numbers(arr):
    '''Return sum of all even numbers in array.'''
    return sum(x for x in arr if x % 2 == 0)

# Example:
# sum_even_numbers([1, 2, 3, 4, 5, 6]) → 12
print(sum_even_numbers([1, 2, 3, 4, 5, 6]))
```"""
                }
            ],
            "strings": [
                {
                    "from": "human",
                    "value": "Write a function that checks if a string is a palindrome, ignoring spaces and case."
                },
                {
                    "from": "gpt",
                    "value": """```python
def is_palindrome(s: str) -> bool:
    '''Check if string is palindrome (ignoring spaces and case).'''
    cleaned = ''.join(s.lower().split())
    return cleaned == cleaned[::-1]

# Examples:
# is_palindrome("Race car") → True
# is_palindrome("hello") → False
print(is_palindrome("Race car"))
```"""
                }
            ],
            "algorithms": [
                {
                    "from": "human",
                    "value": "Implement binary search to find an element in a sorted array."
                },
                {
                    "from": "gpt",
                    "value": """```python
def binary_search(arr, target):
    '''Binary search for target in sorted array. Returns index or -1.'''
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example:
# binary_search([1, 3, 5, 7, 9], 7) → 3
print(binary_search([1, 3, 5, 7, 9], 7))
```"""
                }
            ],
            "oop": [
                {
                    "from": "human",
                    "value": "Create a class representing a BankAccount with deposit, withdraw, and get_balance methods."
                },
                {
                    "from": "gpt",
                    "value": """```python
class BankAccount:
    '''Simple bank account class.'''
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        '''Deposit amount into account.'''
        if amount > 0:
            self.balance += amount
            return True
        return False

    def withdraw(self, amount):
        '''Withdraw amount if sufficient funds.'''
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False

    def get_balance(self):
        '''Return current balance.'''
        return self.balance

# Usage:
account = BankAccount(100)
account.deposit(50)
account.withdraw(30)
print(account.get_balance())  # 120
```"""
                }
            ],
            "file_io": [
                {
                    "from": "human",
                    "value": "Write a function that reads a CSV file and returns the data as a list of dictionaries."
                },
                {
                    "from": "gpt",
                    "value": """```python
import csv

def read_csv_to_dicts(filepath):
    '''Read CSV file and return list of dicts.'''
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        return []

# Example CSV (data.csv):
# name,age,city
# Alice,30,NYC
# Bob,25,LA
#
# read_csv_to_dicts("data.csv") →
# [{'name': 'Alice', 'age': '30', 'city': 'NYC'}, ...]
```"""
                }
            ]
        }

        return {
            "conversations": examples.get(category, examples["arrays"])
        }

    def generate_openclaw_style_example(self, task_type: str = "command_execution") -> Dict:
        """
        Generate OpenClaw-style agent examples

        Args:
            task_type: Type of OpenClaw task (command_execution, debug_recovery, tool_routing, safety_boundary)

        Returns:
            Conversation in OpenClaw format
        """
        if task_type == "command_execution":
            return {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are Clawd. Be safe, concise, and tool-competent. Execute commands efficiently and handle failures gracefully."
                    },
                    {
                        "from": "human",
                        "value": "Need help: Install curl and verify version on Debian. Keep it concise. If command fails, propose a fallback quickly."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll help you install curl on Debian. Let me update the package cache and install curl.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"sudo apt-get update && sudo apt-get install -y curl\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"Hit:1 http://deb.debian.org/debian bookworm InRelease\\nReading package lists... Done\\nThe following NEW packages will be installed:\\n  curl\\n0 upgraded, 1 newly installed, 0 to remove and 0 not upgraded.\\nNeed to get 778 kB of archives.\\nAfter this operation, 2,497 kB of additional disk space will be used.\\nGet:1 http://deb.debian.org/debian bookworm/main amd64 curl amd64 7.88.1-10+deb12u5\\nUnpacking curl ...\\nSetting up curl (7.88.1-10+deb12u5) ...\\nProcessing triggers for man-db ...\\n\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "✓ curl installed successfully. Now verifying version:<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"curl --version\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"curl 7.88.1 (x86_64-pc-linux-gnu)\\nlibcurl/7.88.1 OpenSSL/1.1.1w zlib/1.2.13 brotli/1.0.9 zstd/1.5.4 libidn2/2.3.3 libidn2/0.3.4 libpsl/0.21.2 (+libidn2/2.3.0) libssh2/1.10.0 nghttp2/1.52.0\\nRelease-Date: 2023-10-12\\nProtocols: dict file ftp ftps gopher gophers http https imap imaps ldap ldaps mqtt pop3 pop3s rtsp scp sftp smb smbs smtp smtps telnet tftp \\nFeatures: alt-svc AsynchDNS brotli GSS-API HSTS HTTP2 HTTPS-proxy IPv6 Kerberos Largefile libz NTLM NTLM_WB PSL SPNEGO SSL TLS-SRP UnixSockets zstd\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "✅ **Task Complete**\\n\\n- **curl 7.88.1** installed successfully on Debian\\n- Version verified: `curl 7.88.1 (x86_64-pc-linux-gnu)`\\n- Protocols supported: http, https, ftp, and many more\\n\\nThe installation completed without issues. curl is ready to use."
                    }
                ]
            }
        elif task_type == "debug_recovery":
            return {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are Clawd. Be safe, concise, and tool-competent. Diagnose issues systematically and recover from errors efficiently."
                    },
                    {
                        "from": "human",
                        "value": "nvidia-smi fails with driver communication error; what next? Assume Linux shell. Start with a minimal reproducible check."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll diagnose the nvidia-smi driver communication error systematically. Let me start with basic checks.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"which nvidia-smi\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"/usr/bin/nvidia-smi\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "nvidia-smi is installed. Let me try running it to see the exact error:<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"nvidia-smi 2>&1\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"NVML library initialization failed: Failed to communicate with NVIDIA driver. Make sure that the NVIDIA driver is installed and running.\\nError: Could not communicate with NVIDIA driver.\", \"exit_code\": 1}"
                    },
                    {
                        "from": "gpt",
                        "value": "The error confirms NVML can't communicate with the driver. Let me check if the kernel module is loaded:<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"lsmod | grep nvidia\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "The nvidia kernel module is not loaded. Let me check dmesg for driver errors:<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"dmesg | tail -20 | grep -i nvidia\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"NVRM: API mismatch: the client has version 535.154.05 but the kernel module has version 535.129.06.\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "**🔍 Root Cause Found**: API version mismatch\\n\\n- **Client version**: 535.154.05\\n- **Kernel module**: 535.129.06\\n\\n**Solution**: Reload the kernel module to match versions. Try:\\n\\n```bash\\nsudo rmmod nvidia\\nsudo modprobe nvidia\\n```\\n\\nIf that fails, reboot the system to ensure the correct driver version loads.\\n\\nThe mismatch occurs when the nvidia-smi tool is updated but the old kernel module is still loaded."
                    }
                ]
            }
        elif task_type == "tool_routing":
            return {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are Clawd. Be safe, concise, and tool-competent. Route requests to appropriate tools efficiently."
                    },
                    {
                        "from": "human",
                        "value": "I need to find all Python files in the project that contain the word 'database' and count how many matches there are."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll search for Python files containing 'database' and count the matches. This requires both file finding and content searching.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"find . -name '*.py' -type f -exec grep -l 'database' {} \\; 2>/dev/null\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"./models/database.py\\n./utils/db_helpers.py\\n./tests/test_database.py\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "Found 3 Python files containing 'database'. Now let me count total occurrences:<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"grep -h 'database' ./models/database.py ./utils/db_helpers.py ./tests/test_database.py 2>/dev/null | wc -l\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"47\", \"exit_code\": 0}"
                    },
                    {
                        "from": "gpt",
                        "value": "**🔍 Search Results**\\n\\n- **Files found**: 3 Python files contain 'database'\\n  - `./models/database.py`\\n  - `./utils/db_helpers.py`\\n  - `./tests/test_database.py`\\n- **Total occurrences**: 47 matches\\n\\nThe search efficiently combined `find` for file discovery and `grep` for content matching."
                    }
                ]
            }
        elif task_type == "safety_boundary":
            return {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are Clawd. Be safe, concise, and tool-competent. Respect safety boundaries and refuse dangerous operations."
                    },
                    {
                        "from": "human",
                        "value": "Delete all files in /etc directory to clean up the system configuration."
                    },
                    {
                        "from": "gpt",
                        "value": "❌ **Cannot perform this operation**\\n\\nDeleting files in `/etc` would:**\\n- Destroy critical system configuration\\n- Break the operating system\\n- Likely make the system unbootable\\n- Require complete reinstallation to recover\\n\\n**This is a destructive operation that I cannot perform.**\\n\\nIf you're trying to clean up specific configurations, please:\\n1. Specify which configuration files you want to modify\\n2. Use backup and version control for safe changes\\n3. Consider using a configuration management tool properly\\n\\nFor example, to safely manage a specific service configuration:\\n```bash\\n# Backup first\\nsudo cp /etc/myapp/config.conf /etc/myapp/config.conf.backup\\n# Then edit\\nsudo nano /etc/myapp/config.conf\\n```"
                    }
                ]
            }
        else:  # response_quality
            return {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are Clawd. Be safe, concise, and tool-competent. Provide clear, accurate responses with appropriate detail."
                    },
                    {
                        "from": "human",
                        "value": "What's the difference between a process and a thread in Linux?"
                    },
                    {
                        "from": "gpt",
                        "value": "**Process vs Thread in Linux**\\n\\n**Process** (Heavyweight):\\n- Separate memory space (isolated)\\n- Separate resources (file descriptors, etc.)\\n- Creation: expensive (fork+exec)\\n- Communication: IPC required (pipes, sockets)\\n- Safety: isolated, crash doesn't affect others\\n\\n**Thread** (Lightweight):\\n- Shared memory space (within same process)\\n- Shared resources\\n- Creation: cheap (clone with flags)\\n- Communication: direct memory access\\n- Safety: crash can kill entire process\\n\\n**Key Trade-off**: Threads are faster but less isolated. Processes are isolated but expensive to create.\\n\\n**Example**: Chrome uses processes for tabs (isolation) but threads within each process (parallelism)."
                    }
                ]
            }

    def generate_tool_loop_example(self, complexity: int = 1) -> Dict:
        """
        Generate tool-driven agent loop examples

        Args:
            complexity: 1-3 (number of tools in sequence)

        Returns:
            Conversation with tool interactions
        """
        if complexity == 1:
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Read the file README.md and tell me what it's about."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll read the README.md file for you.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"README.md\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"# My Project\\n\\nThis project does awesome things.\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "The README.md file describes a project called 'My Project' that 'does awesome things'. It appears to be a brief, minimal documentation file."
                    }
                ]
            }
        elif complexity == 2:
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Find all Python files in the current directory and count how many lines of code they contain."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll search for Python files and count their lines.\n<tool_call>\n{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"*.py\", \"path\": \".\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"search_files\", \"content\": \"[\\\"main.py\\\", \\\"utils.py\\\", \\\"config.py\\\"]\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Now I'll read each file to count the lines.<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"main.py\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"def main():\\n    print(\\\"Hello\\\")\\n\\nif __name__ == \\\"__main__\\\":\\n        main()\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Found 3 Python files with a total of approximately 15 lines of code."
                    }
                ]
            }
        else:  # complexity == 3
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Analyze the project structure and create a summary document."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll analyze the project structure systematically.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"find . -type f -name \\\"*.py\\\" | head -10\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"[\\\"./main.py\\\", \\\"./utils.py\\\", \\\"./config.py\\\"]\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Now let me check the git history to understand recent changes.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"git log --oneline -5\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"abc123def Added feature X\\n456789ghi Fixed bug in utils\\ndef345678 Updated README\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "I'll create a summary document based on the analysis.<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"file_path\": \"PROJECT_SUMMARY.md\", \"content\": \"# Project Summary\\n\\n## Structure\\n- main.py: Entry point\\n- utils.py: Utility functions\\n- config.py: Configuration\\n\\n## Recent Changes\\n1. Added feature X\\n2. Fixed bug in utils\\n3. Updated README\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"write_file\", \"content\": \"File written successfully.\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "I've analyzed the project structure and created PROJECT_SUMMARY.md with the findings. The project contains 3 Python files with recent activity including feature additions and bug fixes."
                    }
                ]
            }

    def generate_edge_case_example(self, edge_case_type: str) -> Dict:
        """
        Generate edge case examples

        Args:
            edge_case_type: Type of edge case

        Returns:
            Conversation addressing the edge case
        """
        edge_cases = {
            "empty_file": [
                {
                    "from": "human",
                    "value": "Read the file empty.txt and tell me what it contains."
                },
                {
                    "from": "gpt",
                    "value": "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"empty.txt\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"read_file\", \"content\": \"\"}"
                    },
                {
                    "from": "gpt",
                    "value": "The file empty.txt exists but contains no content (0 bytes)."
                }
            ],
            "command_failure": [
                {
                    "from": "human",
                    "value": "Try to run the command 'invalid_command_xyz' and handle the error gracefully."
                },
                {
                    "from": "gpt",
                    "value": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"invalid_command_xyz\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"bash\", \"content\": \"bash: invalid_command_xyz: command not found\", \"exit_code\": 127}"
                    },
                {
                    "from": "gpt",
                    "value": "The command 'invalid_command_xyz' was not found. The command failed with exit code 127. Would you like me to suggest similar commands or help you find the correct command?"
                }
            ],
            "permission_denied": [
                {
                    "from": "human",
                    "value": "Try to read /etc/shadow and explain what happens."
                },
                {
                    "from": "gpt",
                    "value": "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"/etc/shadow\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"read_file\", \"content\": \"Error: Permission denied\", \"exit_code\": 1}"
                    },
                {
                    "from": "gpt",
                    "value": "Permission was denied when trying to read /etc/shadow. This is expected behavior as /etc/shadow contains sensitive password hash information and is restricted to root users. The system correctly blocked the access attempt for security reasons."
                }
            ],
            "large_output": [
                {
                    "from": "human",
                    "value": "List all files in /usr/lib (there may be thousands). Handle large output."
                },
                {
                    "from": "gpt",
                    "value": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls /usr/lib 2>&1 | head -20\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"bash\", \"content\": \"...showing first 20 of many files...\"}"
                    },
                {
                    "from": "gpt",
                    "value": "The /usr/lib directory contains many files. I've shown the first 20. Due to the large output, I used 'head -20' to limit the display. Would you like me to count the total number of files or search for something specific?"
                }
            ],
            "ambiguous_query": [
                {
                    "from": "human",
                    "value": "Find the file named config (there might be multiple)."
                },
                {
                    "from": "gpt",
                    "value": "I'll search for files named 'config'.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"find . -name \\\"*config*\\\" -type f\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"bash\", \"content\": \"./config.json\\\\n./config.py\\\\n./config.bak\"}"
                    },
                {
                    "from": "gpt",
                    "value": "I found 3 files with 'config' in the name:\\n- ./config.json\\n- ./config.py\\n- ./config.bak\\\\nWhich one would you like me to read?"
                }
            ],
            "syntax_recovery": [
                {
                    "from": "human",
                    "value": "I have a Python file with a syntax error on line 42. Can you help identify and fix it?"
                },
                {
                    "from": "gpt",
                    "value": "I'll help you identify and fix the syntax error. Let me read the file first.<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"broken.py\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"read_file\", \"content\": \"def calculate(x, y):\\n    # Missing colon here\\n    return x + y\"}"
                },
                {
                    "from": "gpt",
                    "value": "I found the syntax error! On line 2, there's a missing colon after the function signature. Here's the fix:<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"file_path\": \"broken.py\", \"content\": \"def calculate(x, y):\\n    # Fixed: added colon\\n    return x + y\"}}\n \n<|im_end|>\n"
                },
                {
                    "from": "tool",
                    "value": "\n{\"name\": \"write_file\", \"content\": \"File written successfully.\"}"
                    },
                {
                    "from": "gpt",
                    "value": "Fixed! The issue was a missing colon on line 2. The function definition should be 'def calculate(x, y):' not 'def calculate(x, y)'. Python requires a colon after function definitions."
                }
            ]
        }

        return {
            "conversations": edge_cases[edge_case_type]
        }

    def generate_skill_invocation_example(self, skill_type: str = "general") -> Dict:
        """
        Generate skill invocation examples (Claude/OpenClaw skills)

        Args:
            skill_type: Type of skill (frontend, commit, testing, research, etc.)

        Returns:
            Conversation showing skill reading and application
        """
        if skill_type == "frontend":
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Create a landing page for a new coffee subscription service called 'BrewDaily'."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll check if there's a frontend design skill available to guide this work.<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \".claude/skills/frontend-design/SKILL.md\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"---\\nname: frontend-design\\ndescription: Create distinctive, production-grade frontend interfaces...\\n---\\n\\nThis skill guides creation of distinctive, production-grade frontend interfaces...\\n\\n## Design Thinking\\nBefore coding, understand the context and commit to a BOLD aesthetic direction...\\n\\n## Frontend Aesthetics Guidelines\\n- Typography: Choose fonts that are beautiful, unique, and interesting\\n- Color & Theme: Commit to a cohesive aesthetic\\n- Motion: Use animations for effects and micro-interactions\\n- Spatial Composition: Unexpected layouts, asymmetry, overlap\\n- Backgrounds & Visual Details: Create atmosphere and depth...\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Perfect! I'll use the frontend-design skill to create a distinctive landing page. Based on the skill guidance, I'll commit to a bold aesthetic direction before coding.\n\n**Design Thinking:**\n- **Purpose**: Convert visitors to coffee subscribers with an appealing, trustworthy interface\n- **Tone**: I'll choose a 'warm, artisanal, retro-futuristic' aesthetic - mixing vintage coffee culture with modern design\n- **Constraints**: Production HTML/CSS/JS, responsive, accessible\n- **Differentiation**: Animated steam rising from coffee cup, parallax scrolling on beans, warm color palette with bold typography\n\nNow implementing with: distinctive display font (Playfair Display), refined body font, warm earth tones with amber accents, CSS animations for steam effects, and generous negative space with strategic overlapping elements.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"file_path\": \"landing.html\", \"content\": \"<!DOCTYPE html>\\n<html>\\n<head>\\n  <link href=\\\"https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lora:wght@400;600&display=swap\\\" rel=\\\"stylesheet\\\">\\n  <style>\\n    :root { --primary: #8B4513; --accent: #FFB347; --cream: #FFF8E7; }\\n    body { font-family: 'Lora', serif; background: var(--cream); }\\n    h1 { font-family: 'Playfair Display', serif; font-size: 4rem; }\\n    .steam { animation: rise 2s infinite; }\\n    @keyframes rise { 0% { opacity: 0; transform: translateY(0); } 100% { opacity: 1; transform: translateY(-50px); } }\\n  </style>\\n</head>\\n<body>\\n  <nav>BrewDaily</nav>\\n  <h1>Fresh Coffee. Daily.</h1>\\n  <button>Start Subscription</button>\\n</body>\\n</html>\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"write_file\", \"content\": \"File written successfully.\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "I've created a distinctive landing page following the frontend-design skill guidance:\n\n**Aesthetic Choices:**\n- **Typography**: Playfair Display (bold, distinctive serif) for headlines, Lora for body\n- **Color**: Warm earth tones with amber accents (#8B4513 primary, #FFB347 accent)\n- **Motion**: CSS steam animation rising from coffee cup\n- **Layout**: Clean, retro-futuristic coffee shop vibe\n- **Atmosphere**: Creamy background (#FFF8E7) evokes warm milk in coffee\n\nThe page avoids generic AI aesthetics (no Inter font, no purple gradients) and commits to a specific artisanal coffee aesthetic. The steam animation adds delightful motion without overwhelming the design."
                    }
                ]
            }
        elif skill_type == "commit":
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "I've made changes to fix the authentication bug. Create a git commit."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll check for commit-related skills to ensure proper git practices.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"find .claude -name \\\"*commit*\" -o -name \\\"*git*\" 2>/dev/null}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \".claude/skills/git-commit/SKILL.md\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Found a git-commit skill! Let me read it.<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \".claude/skills/git-commit/SKILL.md\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"---\\nname: git-commit\\ndescription: Use when user asks to commit changes, create git commits, or mentions commit/push/pull.\\n---\\n\\n## Commit Message Format\\n- Use conventional commits: type(scope): description\\n- Types: feat, fix, docs, style, refactor, test, chore\\n- Body should explain WHY, not WHAT\\n- Include Co-Authored-By trailer for AI assistance\\n\\n## Best Practices\\n- Stage files individually, not git add .\\n- Write commit message in HEREDOC format\\n- Never skip hooks or disable signing\\n- Check git status and diff before committing\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Following the git-commit skill guidance, I'll create a proper conventional commit.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"git status\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"On branch main\\nModified files: auth.py, tests/test_auth.py\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"git diff auth.py\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"diff --git a/auth.py b/auth.py\\n- def authenticate(user, password):\\n+ def authenticate(user, password, max_attempts=3):\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"git add auth.py tests/test_auth.py && git commit -m \\\"$(cat <<'EOF'\\nfix(auth): add rate limiting to prevent brute force attacks\\n\\nAdded max_attempts parameter to authenticate() function to limit\\nauthentication attempts and prevent brute force attacks on user accounts.\\n\\nCo-Authored-By: Claude Sonnet <noreply@anthropic.com>\\nEOF\\n)\\\"\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \"[main abc1234] fix(auth): add rate limiting to prevent brute force attacks\\n 2 files changed, 15 insertions(+), 3 deletions(-)\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "✅ Commit created following the git-commit skill guidelines:\n\n**Format**: Conventional commit `fix(auth):` \n- **Type**: `fix` (bug fix)\n- **Scope**: `(auth)` (authentication module)\n- **Description**: Clear, concise summary\n\n**Body**: Explains WHY (prevent brute force attacks), not just WHAT\n**Trailer**: Includes Co-Authored-By for AI assistance attribution\n\nThe commit message follows best practices with HEREDOC formatting and proper git staging."
                    }
                ]
            }
        else:  # general skill invocation
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Write tests for the user registration module."
                    },
                    {
                        "from": "gpt",
                        "value": "Let me check if there are any testing-related skills available.<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"find .claude/skills -name \\\"*test*\" 2>/dev/null\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"bash\", \"content\": \".claude/skills/test-driven-development/SKILL.md\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Found a TDD skill! Reading it now.<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \".claude/skills/test-driven-development/SKILL.md\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"---\\nname: test-driven-development\\ndescription: Use when user asks to write tests, mentions TDD, testing, or test coverage.\\n---\\n\\n## TDD Workflow\\n1. Write failing test first\\n2. Run test (confirm it fails)\\n3. Write minimal code to pass\\n4. Run test (confirm it passes)\\n5. Refactor\\n\\n## Test Quality\\n- Test behavior, not implementation\\n- Use descriptive test names\\n- Cover edge cases and error conditions\\n- Mock external dependencies\\n- Arrange-Act-Assert pattern\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Following the TDD skill guidance, I'll write comprehensive tests for the user registration module using the Arrange-Act-Assert pattern.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"user_registration.py\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"def register_user(username, email, password):\\n    if not username or not email or not password:\\n        raise ValueError('Missing required fields')\\n    if len(password) < 8:\\n        raise ValueError('Password too short')\\n    # ... registration logic\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"file_path\": \"tests/test_user_registration.py\", \"content\": \"import pytest\\nfrom user_registration import register_user\\n\\nclass TestUserRegistration:\\n    \\\"\\\"\\n    Test suite for user registration functionality.\\n    Following Arrange-Act-Assert pattern and TDD principles.\\n    \\\"\\\"\\n\\n    def test_register_user_with_valid_data(self):\\n        # Arrange\\n        username = \\\"testuser\\\"\\n        email = \\\"test@example.com\\\"\\n        password = \\\"SecurePass123\\\"\\n\\n        # Act\\n        result = register_user(username, email, password)\\n\\n        # Assert\\n        assert result['username'] == username\\n        assert result['email'] == email\\n\\n    def test_register_user_missing_username_raises_error(self):\\n        # Arrange\\n        email = \\\"test@example.com\\\"\\n        password = \\\"SecurePass123\\\"\\n\\n        # Act & Assert\\n        with pytest.raises(ValueError, match='Missing required fields'):\\n            register_user(None, email, password)\\n\\n    def test_register_user_short_password_raises_error(self):\\n        # Arrange\\n        username = \\\"testuser\\\"\\n        email = \\\"test@example.com\\\"\\n        password = \\\"short\\\"\\n\\n        # Act & Assert\\n        with pytest.raises(ValueError, match='Password too short'):\\n            register_user(username, email, password)\\n\\n    def test_register_user_email_validation(self):\\n        # Edge case: invalid email format\\n        with pytest.raises(ValueError, match='Invalid email'):\\n            register_user(\\\"user\\\", \\\"not-an-email\\\", \\\"password123\\\")\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"write_file\", \"content\": \"File written successfully.\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "✅ Created comprehensive test suite following TDD skill guidance:\n\n**Applied TDD Principles:**\n- **Descriptive names**: `test_register_user_with_valid_data` clearly describes what's tested\n- **Behavior over implementation**: Tests validate the API contract, not internal details\n- **Edge cases covered**: Missing fields, short passwords, invalid email format\n- **AAA pattern**: Clear Arrange-Act-Assert structure in each test\n- **Error conditions**: Tests verify proper exception handling\n\n**Test Coverage:**\n- ✅ Valid registration (happy path)\n- ✅ Missing required fields (validation)\n- ✅ Password length validation\n- ✅ Email format validation\n\nRun with: `pytest tests/test_user_registration.py -v`"
                    }
                ]
            }

    def generate_function_calling_example(self, domain: str = "general") -> Dict:
        """
        Generate function calling examples (inspired by Hermes dataset)

        Args:
            domain: Domain of function calling (finance, web, files, etc.)

        Returns:
            Conversation with function calling
        """
        if domain == "finance":
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "What's the current stock price of Apple (AAPL)?"
                    },
                    {
                        "from": "gpt",
                        "value": "I'll fetch the current stock price for Apple.<tool_call>\n{\"name\": \"get_stock_price\", \"arguments\": {\"symbol\": \"AAPL\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"get_stock_price\", \"content\": {\"symbol\": \"AAPL\", \"price\": 178.52, \"change\": \"+2.34\", \"volume\": 52341223}}\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "Apple (AAPL) is currently trading at $178.52, up 2.34% today with a volume of 52.3M shares."
                    }
                ]
            }
        elif domain == "web":
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Search for information about the latest Python release."
                    },
                    {
                        "from": "gpt",
                        "value": "I'll search for the latest Python release information.<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"latest Python release 2026\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"web_search\", \"content\": \"Latest Python release: 3.13.0 released on October 7, 2024...\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "The latest Python release is version 3.13.0, codenamed 'Firmadyads', released on October 7, 2024. It includes new features like improved error messages and performance enhancements."
                    }
                ]
            }
        else:  # general
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": "I need to calculate the sum of numbers in a file called numbers.txt. Can you help?"
                    },
                    {
                        "from": "gpt",
                        "value": "I'll help you calculate the sum of numbers in the file. Let me first read it to see what we're working with.<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"numbers.txt\"}}\n \n<|im_end|>\n"
                    },
                    {
                        "from": "tool",
                        "value": "\n{\"name\": \"read_file\", \"content\": \"42\\n17\\n8\\n23\\n19\"}"
                    },
                    {
                        "from": "gpt",
                        "value": "I can see the file contains the numbers: 42, 17, 8, 23, 19. Let me calculate their sum.\\n\\nSum = 42 + 17 + 8 + 23 + 19 = 109\\n\\nThe sum of all numbers in numbers.txt is 109."
                    }
                ]
            }

    def augment_dataset(self, current_dataset_path: str, output_path: str,
                       num_synthetic: int = 100, strategy: str = "balanced") -> None:
        """
        Augment existing dataset with synthetic examples

        Args:
            current_dataset_path: Path to current training data
            output_path: Path to augmented dataset
            num_synthetic: Number of synthetic examples to add
            strategy: 'balanced' (equal mix) or 'performance' (based on metrics)
        """
        import json

        # Load current dataset
        with open(current_dataset_path, 'r') as f:
            current_data = [json.loads(line) for line in f]

        # Generate synthetic examples
        synthetic_examples = []

        if strategy == "balanced":
            # Equal mix of all example types
            categories = [
                ("one_shot_easy", lambda: self.generate_one_shot_coding_example("easy")),
                ("one_shot_medium", lambda: self.generate_one_shot_coding_example("medium")),
                ("one_shot_hard", lambda: self.generate_one_shot_coding_example("hard")),
                ("codealpaca_arrays", lambda: self.generate_codealpaca_style_example("arrays")),
                ("codealpaca_strings", lambda: self.generate_codealpaca_style_example("strings")),
                ("codealpaca_algorithms", lambda: self.generate_codealpaca_style_example("algorithms")),
                ("codealpaca_oop", lambda: self.generate_codealpaca_style_example("oop")),
                ("codealpaca_fileio", lambda: self.generate_codealpaca_style_example("file_io")),
                ("openclaw_command", lambda: self.generate_openclaw_style_example("command_execution")),
                ("openclaw_debug", lambda: self.generate_openclaw_style_example("debug_recovery")),
                ("openclaw_routing", lambda: self.generate_openclaw_style_example("tool_routing")),
                ("openclaw_safety", lambda: self.generate_openclaw_style_example("safety_boundary")),
                ("tool_loop_1", lambda: self.generate_tool_loop_example(1)),
                ("tool_loop_2", lambda: self.generate_tool_loop_example(2)),
                ("tool_loop_3", lambda: self.generate_tool_loop_example(3)),
                ("edge_case_empty", lambda: self.generate_edge_case_example("empty_file")),
                ("edge_case_error", lambda: self.generate_edge_case_example("command_failure")),
                ("edge_case_permission", lambda: self.generate_edge_case_example("permission_denied")),
                ("skill_invocation_frontend", lambda: self.generate_skill_invocation_example("frontend")),
                ("skill_invocation_commit", lambda: self.generate_skill_invocation_example("commit")),
                ("skill_invocation_general", lambda: self.generate_skill_invocation_example("general")),
                ("function_calling", lambda: self.generate_function_calling_example("general")),
            ]

            examples_per_category = num_synthetic // len(categories)

            for category_name, generator in categories:
                for _ in range(examples_per_category):
                    synthetic_examples.append(generator())

        elif strategy == "performance":
            # Focus on areas needing improvement (would come from benchmark)
            # For now, use balanced
            return self.augment_dataset(current_dataset_path, output_path, num_synthetic, "balanced")

        # Combine datasets
        augmented_data = current_data + synthetic_examples

        # Shuffle
        random.shuffle(augmented_data)

        # Save augmented dataset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for example in augmented_data:
                f.write(json.dumps(example) + '\n')

        print(f"✅ Augmented dataset created:")
        print(f"   Original samples: {len(current_data)}")
        print(f"   Synthetic samples: {len(synthetic_examples)}")
        print(f"   Total samples: {len(augmented_data)}")
        print(f"   Saved to: {output_path}")

def main():
    """Test the synthetic data generator"""
    generator = SyntheticDataGenerator()

    # Test different example types
    print("="*80)
    print("Testing Synthetic Data Generator")
    print("="*80)

    print("\n1. One-shot coding example (medium):")
    example = generator.generate_one_shot_coding_example("medium")
    print(json.dumps(example, indent=2))

    print("\n2. CodeAlpaca-style example (algorithms):")
    example = generator.generate_codealpaca_style_example("algorithms")
    print(json.dumps(example, indent=2))

    print("\n3. Tool loop example (complexity 2):")
    example = generator.generate_tool_loop_example(2)
    print(json.dumps(example, indent=2))

    print("\n4. Edge case example (command failure):")
    example = generator.generate_edge_case_example("command_failure")
    print(json.dumps(example, indent=2))

    print("\n5. Skill invocation example (frontend):")
    example = generator.generate_skill_invocation_example("frontend")
    print(json.dumps(example, indent=2))

    print("\n6. Function calling example:")
    example = generator.generate_function_calling_example("finance")
    print(json.dumps(example, indent=2))

    # Test dataset augmentation
    print("\n" + "="*80)
    print("Testing Dataset Augmentation")
    print("="*80)

    generator.augment_dataset(
        current_dataset_path="./agent-dataset-unsloth/train.jsonl",
        output_path="./agent-dataset-unsloth/train_augmented.jsonl",
        num_synthetic=10,  # Small number for testing
        strategy="balanced"
    )

if __name__ == "__main__":
    main()
