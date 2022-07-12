#include "lib.h"
#include <stdexcept>

int main() {
  // Assume a size of 3x3 for all matrices
  // Implementations in lib should of course generalize to arbitrary size.
  constexpr int N = 3;

  // ========= transpose_in_place =========
  // Initialize 3x3 matrix, assume row major format:
  // 0 1 2
  // 3 4 5
  // 6 7 8
  float matrix[N * N] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  // Call the transpose function
  transpose_in_place(matrix, N);

  // Check that the transpose worked, i.e. we expect:
  // 0 3 6
  // 1 4 7
  // 2 5 8
  if (matrix[1] != 3)
    throw std::runtime_error("transpose_in_place failed :(");

  // ========= transpose_into_buffer =========
  // Initialize 3x3 matrix like before, and another one that we will transpose into
  float matrix1[N * N] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  float matrix2[N * N];

  // Call the transpose function
  transpose_into_buffer(matrix1, matrix2, N);

  // Check that the transpose worked, like above:
  if (matrix2[1] != 3)
    throw std::runtime_error("transpose_into_buffer failed :(");

  // ========= best_hamming_distance =========
  // Find the index of the element in b which has the lowest hamming distance to each element in a
  // i.e. best_idx_i = argmin_j(popcnt(a_i ^ b_j)).
  // If two pairs have the same hamming distance, the one with the lower index must be chosen
  constexpr int M = 3;
  uint8_t a[M] = {0b011101, 0b000001, 0b011100};
  uint8_t b[M] = {0b010001, 0b011101, 0b001101};
  uint8_t best_idxs[M];

  best_hamming_distance(a, b, best_idxs, M);

  // check that he best
  if (best_idxs[1] != 0)
    throw std::runtime_error("best_hamming_distance failed :(");

  return 0;
}
