#ifndef DCEL_HPP
#define DCEL_HPP

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

#include <vector>
#include <fstream>
#include <iostream>

// convenience struct to contain all dcel arrays
struct CuDCEL {
  glm::vec3 *vList; // list of vertex coordinates
  int *vEdgesStart, *vEdgesEnd; // halfedge array slice for each vertex

  int *heSrcList, *heDstList; // list of halfedges
  int *hePairList; // pair index for each halfedge

  int vCount, heCount; // number of vertices/halfedges
};

class DCEL {
  public:
    DCEL(const char *fName, unsigned int blkSize = 128) {
      objRead(fName);
      devInit(blkSize);
    }

    ~DCEL() {
      devFree();
    }

  private:
    std::vector<glm::vec3> vList;
    std::vector<int> heSrcList, heDstList;

    CuDCEL dcel, *dev_dcel;

    void objRead(const char *fName);
    bool objReadVtx(std::istream &fStream);
    bool objReadFace(std::istream &fStream);
    void devInit(int blkDim1d = 256, int blkDim2d = 32);
    void devFree();
};


#endif /* DCEL_HPP */
