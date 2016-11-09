#ifndef DCEL_HPP
#define DCEL_HPP

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

#include <vector>
#include <fstream>
#include <iostream>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "shader.hpp"

template <typename T>
class Bezier;

class DCEL {
  public:
    DCEL(const char *fName);
    ~DCEL();

    void visUpdate();
    void visDraw(Shader *vShader, Shader *tShader);

  private:
    // host data
    std::vector<glm::vec3> vList;
    std::vector<glm::ivec3> triList;
    std::vector<int> fList;
    std::vector<int4> heList;
    std::vector<int2> vBndList;
    std::unordered_map<int, std::vector<int>> loopMap;
    int degMin, degMax;

    cudaGraphicsResource *cuVtxResource;
    glm::vec3 *dev_vList;
    glm::ivec3 *dev_triList;
    int2 *dev_vBndList;
    int4 *dev_heList;
    glm::vec3 *dev_heVtxList;

    Bezier<float> *bezier;
    float *dev_samp;
    float *dev_coeff, *dev_coeff_T;
    float4 *dev_uvGrid;
    int *tessIdx;
    float *dev_tessVtx, *tessVtx;
    float *dev_tessWgt;
    glm::vec3 *dev_tessAllVtx;
    float *dev_tessBez;
    float4 *dev_tessXY;

    int nVtx, nFace, nHe;
    int nSub, nSubFace, nSubVtx;

    // GL visualization data
    float *vboVtxListBuf;
    unsigned int vboIdx, vboVtxList;

    void objRead(const char *fName);
    bool objReadVtx(std::istream &fStream);
    bool objReadFace(std::istream &fStream);
    void heSort();

    void devInit(int blkDim1d = 256, int blkDim2d = 32);
    void devFree();

    void visInit();
    void visFree();

    void sample();
};


#endif /* DCEL_HPP */
