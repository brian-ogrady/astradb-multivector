rm -rf .venv/
uv venv
uv pip install -e ".[all,dev]"
source .venv/bin/activate
python -m unittest tests/test_vector_column_options.py
python -m unittest tests/test_astra_multi_vector_table.py
python -m unittest tests/late_interaction/test_utils.py
#python -m unittest tests/late_interaction/test_base.py
#python -m unittest tests/late_interaction/test_colbert.py
#python -m unittest tests/late_interaction/test_colpali.py
#python -m unittest tests/late_interaction/test_late_interaction_pipeline.py
#python -m unittest tests/test_async_astra_multi_vector_table.py
deactivate
