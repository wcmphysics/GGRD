# GGRD Project Context
This is a Machine Learning project. It relies heavily on scikit-learn for model stacking, PyTorch for neural networks, and Pandas/SciPy for data processing.

## Tech Stack
- Python 3
- Scikit-learn, PyTorch, Pandas, NumPy, SciPy
- Jupyter Notebooks for exploration, Python scripts for utilities.

## Agent Instructions (Rules for Gemini)
1. **Virtual Environment:** Always execute Python commands using the local environment at `\.venv\Scripts\python.exe`.
2. **File Structure:** 
   - Core ML scripts and utilities belong in the `ML/` and `ML/utility/` folders.
   - Sequence-to-Sequence (S2S) models (e.g., PyTorch CNNs) belong in the `S2S/` folder.
   - Do not modify raw input data files (`csv_input.csv`, `excel_test_data.xlsx`).
3. **Code Style:** Use strict PEP-8 formatting. Include type hints and concise docstrings for all new functions.
4. **Jupyter Notebooks:** If you need to edit `.ipynb` files, be careful to preserve the JSON structure, but prefer putting reusable logic into `.py` files inside `ML/utility/` and importing them into the notebooks.