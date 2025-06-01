CXX      := mpicxx
NVCC     := nvcc
CXXFLAGS := -I./include -O2
NVFLAGS  := -I./include -O2

SRCS_CPP := src/main.cpp src/generate_random.cpp src/aggregate.cpp src/globals.cpp
OBJS_CPP := $(SRCS_CPP:.cpp=.o)

SRCS_CU  := src/pi_estimator_kernel.cu
OBJS_CU  := $(SRCS_CU:.cu=.o)

all: monte_carlo_mpi_cuda

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

monte_carlo_mpi_cuda: $(OBJS_CPP) $(OBJS_CU)
	$(CXX) $(CXXFLAGS) $^ -lcudart -o $@

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) monte_carlo_mpi_cuda
