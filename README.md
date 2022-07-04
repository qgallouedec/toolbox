# toolbox

Useful tools

## To upload on PyPI

```bash
pip install wheel
python setup.py sdist
python setup.py bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*
```
