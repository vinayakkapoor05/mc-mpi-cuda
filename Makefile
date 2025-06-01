CXX      := mpicxx
NVCC     := nvcc
CXXFLAGS := -I./include -O2
NVFLAGS  := -I./include -O2

SRCS_CPP := src/main.cpp
OBJS_CPP := $(SRCS_CPP:.cpp=.o)

SRCS_CU  := src/pi_estimator_kernel.cu
OBJS_CU  := $(SRCS_CU:.cu=.o)

all: monte_carlo_mpi_cuda

src/%.o: src/%.cpp
	$(NVCC) -x cu $(CXXFLAGS) --compiler-options -fPIC -ccbin=$(CXX) -c $< -o $@

src/%.o: src/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

monte_carlo_mpi_cuda: $(OBJS_CPP) $(OBJS_CU)
	$(CXX) $(CXXFLAGS) $^ -lcudart -lcurand -o $@

clean:
	rm -f src/*.o monte_carlo_mpi_cuda
