#ifndef PPM_HPP
#define PPM_HPP

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

#include <vector>
#include <fstream>
#include <iostream>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "shader.hpp"

#define PPM_NVARS 6

template <typename T>
class Bezier;

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#endif

struct HeData {
  int32_t src, tgt; // half-edge vertices, source to target
  int32_t deg, ord; // half-edge loop info
  int32_t revIdx;   // index of reverse half-edge (in same list)
  int32_t xIdx;     // index of same half-edge (in other list)
  int32_t bezOff;   // bezier data offset
} MY_ALIGN(32);

class PPM {
  public:
    PPM(bool glVis = true);
    ~PPM();

    void rebuild(const char *fName, int nBasis, int nGrid, int nSub);

    float update(int clickIdx, const glm::vec3 &clickForce, float dt);
    void draw(Shader *vShader);
	
    int intersect(const glm::vec3 &p0, const glm::vec3 &dir);

    // visualization options
    bool visSkel;
    bool visFill;
    bool visDbgNormals;

    // optimization options
    bool useTessSM;
    bool useSampTex;

    // CUDA features
    bool canUseTexObjs;

    // exposed mesh properties
    int nVtx, nFace, nHe;
    int nSub, nSubFace, nSubVtx;
    int degMin, degMax, nDeg;

    // physics attributes
    float kSelf, kNbr, kDamp;
    glm::mat4 model;
  private:
    // PPM properties
    int nBasis, nGrid, nBasis2, nGrid2;
    bool isBuilt;

    // host data
    std::vector<float> vList;
    std::string inFile;
    std::vector<int> fList;
    std::vector<HeData> heLoops, heFaces;
    std::vector<int2> vBndList;
    std::unordered_map<int, std::vector<int>> loopMap;

    // device data
    float *dev_vList;
    int2 *dev_vBndList;
    Bezier<float> *bezier;
    HeData *dev_heLoops, *dev_heFaces;
    float *dev_samp, *dev_coeff;
    float *dev_bezPatch, *dev_wgtPatch;
    float *dev_tessWgt;
    int *dev_tessIdx;
    int *dev_heTessIdx;
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

    // rigid body attributes
    glm::quat rbRot;
    glm::vec3 rbPos;
    glm::vec3 rbAngMom;
    glm::vec3 rbForce, rbTorque;

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
    void physTess();
    void cudaProbe();

    void genSampTex();
    void genCoeff();
    void updateCoeff(int clickIdx, const glm::vec3 &clickForce, float dt);
    void updateRigidBody(int clickIdx, const glm::vec3 &clickForce, float dt);
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
