rm -rf .venv/
uv venv
uv pip install -e ".[all,dev]"
source .venv/bin/activate
python -m unittest tests/test_vector_column_options.py
python -m unittest tests/test_astra_multi_vector_table.py
#python -m unittest tests/test_async_astra_multi_vector_table.py
deactivate
