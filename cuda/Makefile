build: parallel_mult.cpp parallel_mult_kernel.cu
	poetry run python setup.py build
dist: build
	poetry run python setup.py sdist
install: build
	poetry run python setup.py install
clean:
	poetry run python setup.py clean
test: install
	cp parallel_mult.py test/parallel_mult.py
	poetry run pytest test
sanity: install
	poetry run pytest test/basic_test.py
assign: test dist
	mkdir -p assignment
	rm -rf assignment/*
	mkdir -p assignment/src
	mkdir -p assignment/dist
	cp *.cpp assignment/src
	cp *.cu assignment/src
	cp main.py assignment
	cp parallel_mult.py assignment
	cp dist/parallel_mult_cuda-0.1.0.tar.gz assignment/dist
	cp Makefile assignment/src
	cp README.md assignment
	cp requirements.txt assignment

.PHONY: build test install clean
