#ifndef DCEL_HPP
#define DCEL_HPP

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

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
    std::vector<int4> heFaces, heLoops;
    std::vector<int2> vBndList;
    std::unordered_map<int, std::vector<int>> loopMap;

    glm::vec3 *dev_vList;
    glm::ivec3 *dev_triList;
    int2 *dev_vBndList;
    glm::vec3 *dev_heVtxList;

    Bezier<float> *bezier;
    int4 *dev_heLoops, *dev_heFaces;
    float *dev_samp, *dev_coeff;
    float *dev_vtxOut, *vtxOut;
    int *dev_idxOut, *idxOut;
    float2 *dev_bezPatch;
	  float2 *dev_uvIdxMap;
	  int2 *dev_iuvIdxMap;

    // sampling texture
    cudaArray *dev_sampTexArray;
    cudaTextureObject_t sampTexObj = 0;

    int nVtx, nFace, nHe;
    int nSub, nSubFace, nSubVtx;
    int degMin, degMax, nDeg;

    // GL visualization data
    float *vboVtxListBuf;
    unsigned int vboIdx, vboVtxList;

    void objRead(const char *fName);
    bool objReadVtx(std::istream &fStream);
    bool objReadFace(std::istream &fStream);
    void getHeLoops();

    void devInit(int blkDim1d = 256, int blkDim2d = 32);
    void genSampTex();
    void devFree();

    void visInit();
    void visFree();

    void sample();
};


#endif /* DCEL_HPP */
