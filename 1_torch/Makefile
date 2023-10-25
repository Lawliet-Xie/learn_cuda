project := example-app

.PHONY: build
build: configure
	cmake --build build

.PHONY: configure
configure:
	cmake $(project) - B build -DCMAKE_PREFIX_PATH=/content/libtorch

.PHONY: run
run:
	./build/$(project)

.PHONY: clean
clean:
	rm -rf build

#
# 可以使用 Make 来更方便地调用 CMake 命令：
#
#     make build
#     make test
#     make run
#     make clean
#
