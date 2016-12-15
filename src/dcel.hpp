#ifndef DCEL_HPP
#define DCEL_HPP

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
    DCEL(const char *fName, bool glVis = true);
    ~DCEL();

    void rebuild(int nBasis, int nGrid, int nSub);

    float update();
    void draw(Shader *vShader, Shader *tShader);

    // visualization options
    bool visSkel;
    bool visFill;
    bool visDbgNormals;

    // optimization options
    bool useTessSM;
    bool useTessAltSM;
    bool useSampTex;
    bool useSvdUpdate;
    bool useBlasUpdate;

    // functionality flags
    bool canUseTexObjs;
  private:
    // PPM properties
    int nBasis, nGrid, nBasis2, nGrid2;

    // host data
    std::vector<float> vList;
    std::string inFile;
    std::vector<int> fList;
    std::vector<int4> heFaces, heLoops;
    std::vector<int2> vBndList;
    std::unordered_map<int, std::vector<int>> loopMap;

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

    // tessellation info
    int nVtx, nFace, nHe;
    int nSub, nSubFace, nSubVtx;
    int degMin, degMax, nDeg;

    // GL visualization data
    bool useVisualize;
	  unsigned int vboIdx, vboVtx;
	  unsigned int vboTessIdx, vboTessVtx;
	  cudaGraphicsResource *dev_vboTessIdx, *dev_vboTessVtx;

    void objRead(const char *fName);
    bool objReadVtx(std::istream &fStream);
    bool objReadFace(std::istream &fStream);
    void getHeLoops();

    void visInit();
    void devInit();
    void devPatchInit();
    void devMeshInit();
    void devCoeffInit();
    void devTessInit();

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


#endif /* DCEL_HPP */
