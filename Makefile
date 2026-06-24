.PHONY: install test smoke cpp clean

install:
	python -m pip install -e ".[dev,server]"

test:
	python -m pytest -q

smoke:
	python -m cacheir.cli make-tiny examples/tiny_model
	python -m cacheir.cli compile examples/tiny_model --output examples/tiny_cacheir_artifact
	python -m cacheir.cli benchmark examples/tiny_cacheir_artifact --decode-tokens 4 --repeats 1

cpp:
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
	cmake --build cpp/build --config Release

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('cpp/build', ignore_errors=True)"
