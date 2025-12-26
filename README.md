# TileLang LSP (tlsp)

A simple Language Server Protocol (LSP) implementation for TileLang, providing inlay hints for buffer shapes, data types, and memory scopes.

## Features

- **Inlay Hints**: Automatically displays information for `T.alloc_shared`, `T.alloc_local`, `T.alloc_fragment`, `T.match_buffer`, and `T.Tensor` calls.
    - Format: `: shape | dtype | scope`
    - Example: `: (128, 32) | T.float16 | shared`
- **Shape Evaluation**: Attempts to statically evaluate shape arguments, resolving variables and basic arithmetic within the function scope.
- **Dtype & Scope Detection**: Parses data types and memory scopes from arguments or infers them from function names.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Install dependencies:
   ```bash
   pip install pygls
   ```

## Usage

### Running the Server

The server communicates via Stdio. You can run it directly:

```bash
python3 server.py
```

### NeoVim Configuration

To use the server with NeoVim, add the following to your lsp config. It would auto start the server and enable it for Python files.

```lua
vim.lsp.config['tlsp'] = {
cmd = { 'python', 'PATH/TO/tlsp/server.py' },
filetypes = { 'python' },
root_markers = {
  'pyproject.toml',
  'setup.py',
  'setup.cfg',
  'requirements.txt',
  'Pipfile',
  '.git',
},
}
vim.lsp.config['tlsp'].capabilities = vim.lsp.protocol.make_client_capabilities()
vim.lsp.enable("tlsp")
```

<! add a image ----> 
A demo of the server in action: 

![demo](./yq-demo.png)

### VS Code Configuration

TBD; and help is wanted. 

## How it works

The server uses Python's `ast` module to parse the code and traverse the syntax tree. It tracks variable assignments within functions to resolve shape variables used in buffer allocations. It also inspects function arguments to extract data type and scope information.
