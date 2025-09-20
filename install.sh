#!/bin/bash

exec > >(tee last_run.log) 2>&1

# GlycanAnalysisPipeline Installation Script
# Uses Python 3.10 and uv for dependency management

# Parse command line arguments
RUN_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            RUN_MODE=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--run]"
            exit 1
            ;;
    esac
done

# Fix uv installation issues
export UV_LINK_MODE=copy

set -e

echo "Installing GlycanAnalysisPipeline with Python 3.10 and uv..."

# Function to add to PATH if not already present
add_to_path() {
    local dir="$1"
    if [[ ":$PATH:" != *":$dir:"* ]]; then
        export PATH="$dir:$PATH"
        echo "Added $dir to PATH for this session"
        return 0
    fi
    return 1
}

# Check if uv is already installed and accessible
if command -v uv &> /dev/null; then
    echo "uv is already installed and accessible."
else
    # Check if uv exists in ~/.local/bin but PATH issue
    UV_PATH="$HOME/.local/bin/uv"
    if [[ -f "$UV_PATH" ]]; then
        echo "uv found at $UV_PATH but not in PATH. Adding to PATH..."
        if add_to_path "$HOME/.local/bin"; then
            if command -v uv &> /dev/null; then
                echo "uv is now accessible!"
            else
                echo "Still unable to access uv after adding to PATH."
                exit 1
            fi
        else
            echo "uv already in PATH but still not accessible. Please check permissions."
            exit 1
        fi
    else
        echo "Installing uv package manager..."
        # Download and execute the installation script properly
        curl -LsSf https://astral.sh/uv/install.sh | bash -s --
        
        # Add ~/.local/bin to PATH
        if add_to_path "$HOME/.local/bin"; then
            echo "uv installation complete!"
        else
            echo "uv was already in PATH after installation."
        fi
        
        # Double-check if uv is now accessible
        if ! command -v uv &> /dev/null; then
            echo "ERROR: uv is still not accessible after installation."
            echo "Trying alternative installation method..."
            
            # Alternative: Install via pip if possible
            if command -v python3 &> /dev/null; then
                echo "Attempting to install uv via pip..."
                python3 -m pip install --user uv
                if add_to_path "$HOME/.local/bin"; then
                    if command -v uv &> /dev/null; then
                        echo "uv installed successfully via pip!"
                    else
                        echo "ERROR: uv installation failed via both methods."
                        echo "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
                        exit 1
                    fi
                fi
            else
                echo "ERROR: No Python available for pip installation."
                echo "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
                exit 1
            fi
        fi
    fi
fi

echo "uv version: $(uv --version)"

# Install Python 3.10 if not available
echo "Checking Python 3.10 installation..."
if ! uv python list | grep -q "3.10"; then
    echo "Installing Python 3.10..."
    uv python install 3.10
fi

# Create virtual environment with Python 3.10
echo "Creating virtual environment with Python 3.10..."
# Set UV_VENV_CLEAR=1 when in run mode to automatically replace existing env without prompting
if [[ "$RUN_MODE" == true ]]; then
    export UV_VENV_CLEAR=1
    echo "Running in --run mode: will automatically replace existing virtual environment if present."
    
    # Check if .venv exists and is valid (has requirements.txt installed)
    if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
        # Check if requirements are installed by looking for a common dependency
        if [[ -f ".venv/bin/python" && -f "requirements.txt" ]]; then
            # Check if pip is available and list installed packages briefly
            if ./.venv/bin/python -c "import sys; sys.exit(0)" 2>/dev/null; then
                # Quick check for a key dependency (e.g., polars from requirements.txt)
                if ./.venv/bin/python -c "import polars; print('requirements ok')" 2>/dev/null; then
                    echo "Virtual environment exists and requirements are installed. Activating existing environment."
                    source .venv/bin/activate
                    ENV_EXISTS=true
                else
                    echo "Virtual environment exists but requirements are missing. Recreating environment."
                    rm -rf .venv
                    uv venv --python 3.10 .venv
                    source .venv/bin/activate
                    ENV_EXISTS=false
                fi
            else
                echo "Virtual environment exists but Python is not functional. Recreating environment."
                rm -rf .venv
                uv venv --python 3.10 .venv
                source .venv/bin/activate
                ENV_EXISTS=false
            fi
        else
            echo "Virtual environment directory exists but is incomplete. Recreating environment."
            rm -rf .venv
            uv venv --python 3.10 .venv
            source .venv/bin/activate
            ENV_EXISTS=false
        fi
    else
        echo "No valid virtual environment found. Creating new environment."
        uv venv --python 3.10 .venv
        source .venv/bin/activate
        ENV_EXISTS=false
    fi
else
    echo "If a virtual environment already exists, you'll be prompted to replace it."
    uv venv --python 3.10 .venv
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
fi

# Install dependencies using uv
echo "Installing dependencies with uv..."
if [[ "$RUN_MODE" == true && "$ENV_EXISTS" == true ]]; then
    echo "Requirements already installed in existing environment. Skipping installation."
else
    # Use --force-reinstall when in run mode to ensure fresh installation, otherwise normal install
    if [[ "$RUN_MODE" == true ]]; then
        uv pip install --force-reinstall -r requirements.txt
    else
        uv pip install -r requirements.txt
    fi
fi

echo "Installation complete!"
echo "Virtual environment created at .venv"

# Check if environment variables are set
echo "Checking environment variables..."

REQUIRED_VARS=(
    "GLYCOSHAPE_DATA_DIR"
    "GLYCOSHAPE_PROCESS_DIR" 
    "GLYCOSHAPE_OUTPUT_PATH"
    "GLYCOSHAPE_INVENTORY_PATH"
    "GLYTOUCAN_CONTRIBUTOR_ID"
    "GLYTOUCAN_API_KEY"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        MISSING_VARS+=("$var")
    fi
done

if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
    echo "WARNING: Missing required environment variables: ${MISSING_VARS[*]}"
    echo "Please set these variables before running the pipeline."
    echo "See README.md for configuration instructions."
    echo ""
    if [[ "$RUN_MODE" == true ]]; then
        echo "Running in --run mode: Continuing with test run despite missing variables."
    else
        echo "Would you like to continue with a test run anyway? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Installation script completed. Please configure environment variables and run 'python main.py' manually."
            exit 0
        fi
    fi
else
    echo "All required environment variables are set."
fi

# Run main.py to verify installation
echo ""
echo "Running main.py to verify installation..."
echo "This may take some time depending on your data..."

# Run main.py and capture output
if python main.py; then
    echo ""
    echo "✓ Installation verification successful!"
    echo "The pipeline ran successfully."
    echo ""
    echo "To run the pipeline again in the future:"
    echo "1. Activate environment: source .venv/bin/activate"
    echo "2. Run: python main.py"
else
    echo ""
    echo "✗ Pipeline run completed with errors."
    echo "Installation was successful, but the pipeline encountered issues."
    echo "Please check your data files and environment variables."
    echo "To troubleshoot, activate the environment and run 'python main.py' manually."
    exit 1
fi

# Make the script executable (for Linux/macOS)
chmod +x install.sh

echo ""
echo "Setup complete! Your GlycanAnalysisPipeline is ready to use."

# Suggest permanent PATH update if needed
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "Tip: To make uv available in future sessions, add this to your ~/.bashrc or ~/.zshrc:"
    echo 'export PATH="$HOME/.local/bin:$PATH"'
    echo "Then run: source ~/.bashrc"
fi

echo ""
echo "To activate the environment manually: source .venv/bin/activate"
echo "To run the pipeline manually: python main.py"
echo ""
echo "Windows users: Use '.venv\Scripts\activate' instead of 'source .venv/bin/activate'"
echo ""
