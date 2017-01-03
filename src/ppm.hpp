#ifndef PPM_HPP
#define PPM_HPP

#define _USE_MATH_DEFINES
#include <cmath>

#include <vector>
#include <fstream>
#include <iostream>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifdef __CUDA_ARCH__
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

#include "shader.hpp"

#define PPM_NVARS 6

template <typename T>
class Bezier;

class PPM {
  public:
    PPM(bool glVis = true);
    ~PPM();

    void rebuild(const char *fName, int nBasis, int nGrid, int nSub);

    float update();
    void draw(Shader *vShader, Shader *tShader);

    // visualization options
    bool visSkel;
    bool visFill;
    bool visDbgNormals;

    // optimization options
    bool useTessSM;
    bool useSampTex;

    // functionality flags
    bool canUseTexObjs;
    
    // exposed mesh properties
    int nVtx, nFace, nHe;
    int nSub, nSubFace, nSubVtx;
    int degMin, degMax, nDeg;
  private:
    // PPM properties
    int nBasis, nGrid, nBasis2, nGrid2;
    bool isBuilt;

    // host data
    std::vector<float> vList;
    std::string inFile;
    std::vector<int> fList;
    std::vector<int4> heFaces, heLoops;
    std::vector<int2> vBndList;
    std::unordered_map<int, std::vector<int>> loopMap;

    // device data
    float *dev_vList;
    int2 *dev_vBndList;
    Bezier<float> *bezier;
    int4 *dev_heLoops, *dev_heFaces;
    float *dev_samp, *dev_coeff;
    float2 *dev_bezPatch;
    float2 *dev_uvIdxMap;
    int2 *dev_iuvIdxMap;
    float *dev_tessWgt;
    int *dev_tessIdx;
    float *dev_tessVtx;
    float *dev_dv;

    // sampling texture
    cudaArray *dev_sampTexArray;
    cudaTextureObject_t sampTexObj;

    // GL visualization data
    bool useVisualize;
    unsigned int vaoBase, vaoTess;
	  unsigned int vboIdx, vboVtx;
	  unsigned int vboTessIdx, vboTessVtx;
	  cudaGraphicsResource *dev_vboTessIdx, *dev_vboTessVtx;
    
    // physics data
    glm::mat3 moi;
    glm::vec3 cm;
    float mass;

    bool objRead(const char *fName);
    bool objReadVtx(std::istream &fStream);
    bool objReadFace(std::istream &fStream);
    void getHeLoops();

    void visInit();
    void devInit();
    void devPatchInit();
    void devMeshInit();
    void devCoeffInit();
    void devTessInit();
    void physInit();

    void genSampTex();
    void genCoeff();
    void updateCoeff();
    
    std::vector<void*> allocList;
    template <typename T>
    void devAlloc(T **ptr, size_t size) {
      cudaMalloc(ptr, size);
      allocList.push_back(*ptr);
    }

    void devFree();
    void visFree();
    void sample();
};


#endif /* PPM_HPP */
