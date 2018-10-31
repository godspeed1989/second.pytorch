#include <stdio.h>
#include <array>
#include <vector>
#include <unordered_map>
#include <google/dense_hash_map> // sparsehash

using Int = int64_t;

/** mp operations
 * set_empty_key()
 */

// Point<dimension> is a point in the d-dimensional integer lattice
// (i.e. square-grid/cubic-grid, ...)
template <Int dimension> using Point = std::array<Int, dimension>;

// FNV Hash function for Point<dimension>
template <Int dimension> struct IntArrayHash {
  std::size_t operator()(Point<dimension> const &p) const {
    Int hash = -3750763034362895579; // 14695981039346656037;
    for (auto x : p) {
      hash *= 1099511628211;
      hash ^= x;
    }
    return hash;
  }
};

// Point -> Int
template <Int dimension>
using SparseGridMap =
    google::dense_hash_map<Point<dimension>, Int,
                           IntArrayHash<dimension>,
                           std::equal_to<Point<dimension>>>;
template <Int dimension> class SparseGrid {
public:
  Int ctr;
  SparseGridMap<dimension> mp;
  SparseGrid();
};

template <Int dimension> SparseGrid<dimension>::SparseGrid() : ctr(0) {
  // Sparsehash needs a key to be set aside and never used
  Point<dimension> empty_key;
  for (Int i = 0; i < dimension; ++i)
    empty_key[i] = std::numeric_limits<Int>::min();
  mp.set_empty_key(empty_key);
}

template <Int dimension> using SparseGrids =
    std::vector<SparseGrid<dimension>>;

// Hash tables for each scale locating the active points
template <Int dimension> using Girds =
    std::unordered_map<Point<dimension>, SparseGrids<dimension>, IntArrayHash<dimension>>;

int main()
{
  Point<2> p;
  p[0] = 1;
  p[1] = 2;
  SparseGrid<2> sg;
  auto iter = sg.mp.find(p);
  if (iter == sg.mp.end()) {
    printf("not found\n");
  }
  p[0] = 1;   p[1] = 1;
  sg.mp.insert(std::make_pair(p, 0));
  p[0] = 2;   p[1] = 2;
  sg.mp.insert(std::make_pair(p, 1));
  for (auto it = sg.mp.begin(); it != sg.mp.end(); ++it) {
    printf("(%d, %d) -> %d\n", it->first[0], it->first[1], it->second);
  }

  Girds<2> grids;
  grids.clear();

  return 0;
}
