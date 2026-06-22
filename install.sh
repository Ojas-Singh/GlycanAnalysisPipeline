#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

resolve_log_path() {
    if [[ -n "$GLYCOSHAPE_GAP_LOG" ]]; then
        printf '%s\n' "$GLYCOSHAPE_GAP_LOG"
        return
    fi

    if [[ -f ".env" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            [[ -z "$line" || "$line" =~ ^# ]] && continue
            line=${line#export }

            if [[ "$line" != GLYCOSHAPE_GAP_LOG=* ]]; then
                continue
            fi

            value=${line#*=}
            value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            if [[ "$value" =~ ^\".*\"$ || "$value" =~ ^\'.*\'$ ]]; then
                value=${value:1:${#value}-2}
            fi

            printf '%s\n' "$value"
            return
        done < .env
    fi

    printf '%s\n' "last_run.log"
}

LOG_PATH="$(resolve_log_path)"
mkdir -p "$(dirname "$LOG_PATH")"
if [[ "${GLYCOSHAPE_NO_TEE:-}" != "1" ]]; then
    exec > >(tee "$LOG_PATH") 2>&1
fi

# GlycanAnalysisPipeline Installation Script
# Uses Python 3.10 and uv for dependency management

# Parse command line arguments
RUN_MODE=false
BACKGROUND_MODE=false
MAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            RUN_MODE=true
            shift
            ;;
        --background|--nohup)
            BACKGROUND_MODE=true
            shift
            ;;
        --)
            shift
            MAIN_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--run] [--background|--nohup] [-- <main.py args>]"
            exit 1
            ;;
    esac
done

if [[ "$BACKGROUND_MODE" == true ]]; then
    PID_PATH="$(dirname "$LOG_PATH")/GlycoShape_GAP.pid"
    RUNNER_PATH="$(dirname "$LOG_PATH")/GlycoShape_GAP.run.sh"
    CHILD_ARGS=()
    if [[ "$RUN_MODE" == true ]]; then
        CHILD_ARGS+=("--run")
    fi
    if [[ ${#MAIN_ARGS[@]} -gt 0 ]]; then
        CHILD_ARGS+=("--" "${MAIN_ARGS[@]}")
    fi

    : > "$LOG_PATH"
    {
        echo "#!/bin/bash"
        printf 'cd %q\n' "$SCRIPT_DIR"
        echo "export GLYCOSHAPE_NO_TEE=1"
        printf 'exec /bin/bash %q' "$SCRIPT_DIR/install.sh"
        for arg in "${CHILD_ARGS[@]}"; do
            printf ' %q' "$arg"
        done
        printf '\n'
    } > "$RUNNER_PATH"
    chmod +x "$RUNNER_PATH"

    if command -v setsid >/dev/null 2>&1; then
        nohup setsid "$RUNNER_PATH" >> "$LOG_PATH" 2>&1 &
    else
        nohup "$RUNNER_PATH" >> "$LOG_PATH" 2>&1 &
    fi
    bg_pid=$!
    disown "$bg_pid" 2>/dev/null || true
    printf '%s\n' "$bg_pid" > "$PID_PATH"
    echo "Started GlycanAnalysisPipeline in background."
    echo "PID: $bg_pid"
    echo "PID file: $PID_PATH"
    echo "Log: $LOG_PATH"
    exit 0
fi

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

# Ensure Java is installed (required for running java -jar tools)
echo "Checking Java installation..."
export_java_env() {
    local java_home="$1"
    export JAVA_HOME="$java_home"
    export PATH="$JAVA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$JAVA_HOME/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
}

java_works() {
    local java_bin="$1"
    [[ -x "$java_bin" ]] && "$java_bin" -version >/dev/null 2>&1
}

ensure_java() {
    if command -v java >/dev/null 2>&1; then
        if [[ -z "${JAVA_HOME:-}" ]]; then
            local detected_java_bin
            detected_java_bin="$(command -v java)"
            if [[ -n "$detected_java_bin" ]]; then
                detected_java_bin="$(readlink -f "$detected_java_bin" 2>/dev/null || echo "$detected_java_bin")"
                if [[ "$detected_java_bin" == */bin/java ]]; then
                    export_java_env "$(cd "$(dirname "$detected_java_bin")/.." && pwd)"
                fi
            fi
        else
            export_java_env "$JAVA_HOME"
        fi
        if java -version >/dev/null 2>&1; then
            echo "Java detected: $(java -version 2>&1 | head -n 1)"
            return 0
        fi
        echo "Detected Java is unusable; installing a local JRE"
    fi

    local default_java_home="$SCRIPT_DIR/.java"
    local java_home="${JAVA_HOME:-$default_java_home}"
    local java_bin="$java_home/bin/java"

    if [[ ! -x "$java_bin" && "$java_home" == "$default_java_home" ]]; then
        local discovered_java_bin
        discovered_java_bin="$(find "$default_java_home" -mindepth 2 -maxdepth 3 -path '*/bin/java' -type f -perm -u+x 2>/dev/null | head -n 1 || true)"
        if [[ -n "$discovered_java_bin" ]]; then
            java_home="$(cd "$(dirname "$discovered_java_bin")/.." && pwd)"
            java_bin="$discovered_java_bin"
        fi
    fi

    if java_works "$java_bin"; then
        export_java_env "$java_home"
        echo "Using locally installed Java at $JAVA_HOME"
        return 0
    elif [[ -x "$java_bin" && "$java_home" == "$default_java_home"* ]]; then
        echo "Removing incomplete local Java installation at $java_home"
        rm -rf "$default_java_home"
        java_home="$default_java_home"
        java_bin="$java_home/bin/java"
    fi

    if [[ -n "$JAVA_HOME" && ! -w "$java_home" ]]; then
        echo "Configured JAVA_HOME=$java_home is not writable; falling back to $default_java_home"
        java_home="$default_java_home"
        java_bin="$java_home/bin/java"
    fi

    echo "Java not found. Installing a local JRE under $java_home without sudo..."
    rm -rf "$java_home"
    mkdir -p "$java_home"
    export JAVA_HOME="$java_home"

    local installed_java_home
    installed_java_home="$(python - <<'PY'
import os
import sys
import ssl

try:
    import jdk  # type: ignore
except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1)

java_home = os.environ["JAVA_HOME"]
ssl._create_default_https_context = ssl._create_unverified_context
installed = jdk.install("17", jre=True, path=java_home)
print(installed)
PY
)"

    if [[ -n "$installed_java_home" && -x "$installed_java_home/bin/java" ]]; then
        export_java_env "$installed_java_home"
        echo "Local Java installation ready: $(java -version 2>&1 | head -n 1)"
        return 0
    fi

    echo "ERROR: Local Java installation failed."
    return 1
}

ensure_java

# Function to load .env file
load_dotenv() {
    if [[ -f ".env" ]]; then
        echo "Loading environment variables from .env file..."
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Trim leading/trailing whitespace
            line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            # Skip comments and empty lines
            [[ -z "$line" || "$line" =~ ^# ]] && continue
            # Support optional 'export ' prefix
            line=${line#export }
            # Ensure the line contains an '=' for KEY=VALUE
            if [[ "$line" != *"="* ]]; then
                echo "Skipping invalid .env line (no '='): $line" >&2
                continue
            fi
            # Split on first '=' only to allow '=' inside values
            key=${line%%=*}
            value=${line#*=}
            # Trim whitespace from key/value
            key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            # Validate key (alnum and underscore, cannot start with digit)
            if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
                echo "Skipping invalid env var name: $key" >&2
                continue
            fi
            # Remove optional surrounding quotes from value
            if [[ "$value" =~ ^\".*\"$ || "$value" =~ ^\'.*\'$ ]]; then
                value=${value:1:${#value}-2}
            fi
            # Set only if not already set in environment
            if [[ -z "${!key}" ]]; then
                export "$key=$value"
            fi
        done < .env
    fi
}

# Load .env file if present
load_dotenv

# Check if environment variables are set
echo "Checking environment variables..."

# Backward compatibility: support GLYCOSHAPE_OUTPUT_PATH if set
if [[ -z "$GLYCOSHAPE_OUTPUT_DIR" && -n "$GLYCOSHAPE_OUTPUT_PATH" ]]; then
    export GLYCOSHAPE_OUTPUT_DIR="$GLYCOSHAPE_OUTPUT_PATH"
fi

REQUIRED_VARS=(
    "GLYCOSHAPE_DATA_DIR"
    "GLYCOSHAPE_PROCESS_DIR" 
    "GLYCOSHAPE_OUTPUT_DIR"
    "GLYTOUCAN_CONTRIBUTOR_ID"
    "GLYTOUCAN_API_KEY"
)

POCKETBASE_EFFECTIVE_TOKEN="${POCKETBASE_TOKEN:-${POCKETBASE_ADMIN_TOKEN:-}}"
if [[ -z "$POCKETBASE_URL" || -z "$POCKETBASE_EFFECTIVE_TOKEN" ]]; then
    REQUIRED_VARS+=("GLYCOSHAPE_INVENTORY_PATH")
else
    echo "PocketBase submission metadata enabled via POCKETBASE_URL and token."
fi

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        MISSING_VARS+=("$var")
    fi
done

if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
    echo "WARNING: Missing required environment variables: ${MISSING_VARS[*]}"
    echo "Please set these variables before running the pipeline."
    echo "You can set them in your shell environment or in a .env file in the project root."
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
if python main.py "${MAIN_ARGS[@]}"; then
    echo ""
    echo "✓ Installation verification successful!"
    echo "The pipeline ran successfully."
    echo ""
    echo "To run the pipeline again in the future:"
    echo "1. Activate environment: source .venv/bin/activate"
    echo "2. Run: python main.py ${MAIN_ARGS[*]}"
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
