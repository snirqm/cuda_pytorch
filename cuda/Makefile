build: parallel_mult.cpp parallel_mult_kernel.cu
	poetry run python setup.py build
dist: build
	poetry run python setup.py sdist
install: build
	cp parallel_mult.py test/parallel_mult.py
	poetry run python setup.py install
clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf assignment
	poetry run python setup.py clean

uninstall:
	pip uninstall -y parallel_mult_cuda

test: uninstall install
	cp parallel_mult.py test/parallel_mult.py
	poetry run pytest test --timeout=1 --timeout_method=thread

sanity: uninstall install
	poetry run pytest test/basic_test.py --timeout=1 --timeout_method=thread

assign: clean test dist
	mkdir -p assignment
	rm -rf assignment/*
	mkdir -p assignment/src
	mkdir -p assignment/dist
	cp *.cpp assignment/src
	cp *.cu assignment/src
	cp main.py assignment
	cp parallel_mult.py assignment
	cp dist/parallel_mult_cuda-*.*.*.tar.gz assignment/dist
	cp Makefile assignment/src
	cp README.md assignment
	cp requirements.txt assignment
	cd assignment
	pip install dist/parallel_mult_cuda-*.*.*.tar.gz
	python main.py
	cd -

.PHONY: build test install clean
