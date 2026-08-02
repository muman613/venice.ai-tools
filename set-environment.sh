
VENV_PATH="$HOME/venv"
VENICE_KEY_PATH="$HOME/bin/venice.env"

function set_venv() {
    echo "Using virtual environment... ($VENV_PATH)"
    source "$VENV_PATH/bin/activate"
}

function set_venice_key() {
    echo "Setting venice.ai key..."
    source "$VENICE_KEY_PATH"
}

function set_environment() {
    set_venv
    set_venice_key
}

set_environment
