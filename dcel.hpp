#ifndef DCEL_HPP
#define DCEL_HPP

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

#include <vector>
#include <fstream>
#include <iostream>

#include "shader.hpp"

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
    DCEL(const char *fName);
    ~DCEL();

    void visUpdate();
    void visDraw(Shader *shader);

  private:
    // host-friendly data
    std::vector<glm::vec3> vList;
    std::vector<glm::vec3> nrmList;
    std::vector<int> fList;
    std::vector<int> heSrcList, heDstList;

    CuDCEL dcel, *dev_dcel;

    // GL visualization data
    float *vboDataBuf;
    unsigned int vboData, vboIdx;
    int ssoVtx, ssoNrm;

    void objRead(const char *fName);
    bool objReadVtx(std::istream &fStream);
    bool objReadFace(std::istream &fStream);

    void devInit(int blkDim1d = 256, int blkDim2d = 32);
    void devFree();

    void visInit();
};


#endif /* DCEL_HPP */
