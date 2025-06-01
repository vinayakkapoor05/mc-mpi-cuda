#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

void generate_random_numbers();
void monte_carlo_kernel(int rank, int size);
void aggregate(int rank, int size);

#endif // MONTE_CARLO_H