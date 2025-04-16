PYTHON=python3
VENV_DIR=.venv
ACTIVATE=. $(VENV_DIR)/bin/activate

init:
	@echo "Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Environment ready. Run 'source $(VENV_DIR)/bin/activate' to activate it."

run:
	@echo "Launching Streamlit app..."
	$(ACTIVATE) && streamlit run streamlit_app.py

index:
	@echo "Building FAISS index from documents..."
	$(ACTIVATE) && $(PYTHON) rag/build_index.py

