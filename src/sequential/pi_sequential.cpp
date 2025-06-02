#include <iostream>
#include <random>
#include <chrono>

int main() {
    // same number of points/trials as parallel (MPI + CUDA)
    const long long NUM_POINTS = 400000000LL;
    
    std::random_device rd;
    // initialize mt19937_64 random number generator
    std::mt19937_64 rng(rd());

    // produce random points
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();

    // check to see whether the point lies inside the circle or not
    long long inside = 0;
    for (long long i = 0; i < NUM_POINTS; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        if (x*x + y*y <= 1.0) {
            ++inside;
        }
    }

    // compute the pi estimate
    double pi_estimate = 4.0 * static_cast<double>(inside) / static_cast<double>(NUM_POINTS);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_taken = end - start;

    std::cout << "Pi ≈ " << pi_estimate << std::endl;
    std::cout << "Time Taken: " << time_taken.count() << " seconds" << std::endl;
    return 0;
}
