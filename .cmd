autoflake --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --in-place --recursive .
isort .
black .

set NUMBA_DISABLE_JIT=1
coverage run -m unittest discover tests
coverage xml
