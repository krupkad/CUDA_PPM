#include "ppm.hpp"
#include "shader.hpp"
#include "util/error.hpp"

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <algorithm>


PPM::PPM(bool glVis) :
  useTessSM(false),
  canUseTexObjs(true),
  useSampTex(true),
  useVisualize(glVis),
  visFill(true),
  visSkel(true),
  visDbgNormals(true),
  inFile(""),
  isBuilt(false),
  kSelf(10.0f),
  kDamp(1.5f),
  kNbr(10.0f)
{}

PPM::~PPM() {
  if (isBuilt) {
    devFree();
    visFree();
  }
}

void PPM::rebuild(const char *fName, int nBasis, int nGrid, int nSub) {
  if (isBuilt) {
    vList.clear();
    vBndList.clear();
    heFaces.clear();
    heLoops.clear();
    fList.clear();
    loopMap.clear();
    visFree();
    devFree();
    isBuilt = false;
  }
  
  if (!objRead(fName)) {
    fprintf(stderr, "couldn't open file %s\n", fName);
    return;
  }
  inFile = fName;
  fprintf(stderr, "read done\n");

  // tesselation controls
  this->nSub = nSub;
  this->nSubFace = nSub*nSub;
  this->nSubVtx = (nSub + 1)*(nSub + 2) / 2;

  // check CUDA features
  cudaProbe();

  // initialize device-side mesh data
  devMeshInit();

  // calculate mesh physics properties
  physInit();

  // initialize visualization data
  if (useVisualize) {
    visInit();
    fprintf(stderr, "vis done\n");
  }
 
  // calculate PPM data 
  this->nBasis = nBasis;
  this->nGrid = nGrid;
  this->nBasis2 = nBasis*nBasis;
  this->nGrid2 = nGrid*nGrid;
  devPatchInit();
  devCoeffInit();

  // tesselate mesh
  devTessInit();

  fprintf(stderr, "dev done\n");
  isBuilt = true;
}

// sort half edges a) by source, b) in patch boundary order
void PPM::getHeLoops() {
  for(auto &itr : loopMap) {
    std::vector<int> &loop = itr.second;

    // check if there's a vertex with no predecessor i.e.
    // we're on a boundary, and make it first
    int first;
    for (first = 0; first < loop.size(); first++) {
      bool hasPred = false;
      for (int i = first+1; i < loop.size(); i++) {
        if (heFaces[loop[first]].src == heFaces[loop[i]].tgt) {
          hasPred = true;
          break;
        }
      }

      if (!hasPred)
        break;
    }
    if (first != loop.size()) 
      std::swap(loop[first],loop[0]);

    // finish the chain
    for (int i = 0; i < loop.size()-1; i++) {
      int dst = heFaces[loop[i]].tgt;
      for (int j = i+1; j < loop.size(); j++) {
        int src = heFaces[loop[j]].src;
        if (src == dst) {
          std::swap(loop[i+1],loop[j]);
          break;
        }
      }

    }
  }

  // create a halfedge list ordered by both source vertex and loop sequence
  degMin = INT_MAX;
  degMax = INT_MIN;
  for (int v1 = 0; v1 < nVtx; v1++) {
    int2 vRange;
    vRange.x = heLoops.size();
    struct HeData he;
    for (int i = 0; i < loopMap[v1].size(); i++) {
      int heIdx = loopMap[v1][i];
      he.src = v1;
      he.tgt = heFaces[heIdx].src;
      int fIdx = heIdx/3;
      for (int k = 0; k < 3; k++) {
        if (heFaces[3*fIdx+k].src == he.src && heFaces[3*fIdx+k].tgt == he.tgt)
          heIdx = 3*fIdx+k;
      }
      he.xIdx = heIdx;
      heFaces[heIdx].xIdx = heLoops.size();
      he.ord = heFaces[heIdx].ord = i;
      he.deg = heFaces[heIdx].deg = loopMap[v1].size();
      heLoops.push_back(he);
    }
	  vRange.y = heLoops.size();
    vBndList.push_back(vRange);

    if (vRange.y - vRange.x < degMin)
      degMin = vRange.y - vRange.x;
    if (vRange.y - vRange.x > degMax)
      degMax = vRange.y - vRange.x;
  }
  nDeg = degMax - degMin + 1;
}

bool PPM::objReadVtx(std::istream &fStream) {
  float p;
  for (int i = 0; i < 3; i++) {
    fStream >> p;
    vList.push_back(p);
  }
  for (int i = 0; i < 3; i++)
    vList.push_back(0.0f);

  return true;
}

bool PPM::objReadFace(std::istream &fStream) {
  std::string vDesc;
  std::vector<int> vIdxList;
  while (!fStream.eof()) {
    fStream >> vDesc;

    int tokLen = vDesc.find('/');
    if (tokLen == -1)
      tokLen = vDesc.size();
    if (tokLen == 0)
      return false;

    int vIdx;
    if (sscanf(vDesc.substr(0,tokLen).c_str(), "%d", &vIdx) < 1)
      return false;

    if (vIdx >= 0)
      vIdxList.push_back(vIdx);
    else
      vIdxList.push_back(vList.size()/PPM_NVARS + vIdx + 1);
  }
  unsigned int N = vIdxList.size();

  if (N != 3) {
    fprintf(stderr, "nontriangular face\n");
    return false;
  }

  HeData he;

  he.src = vIdxList[0]-1;
  he.tgt = vIdxList[1]-1;
  fList.push_back(vIdxList[0] - 1);
  loopMap[vIdxList[2]-1].push_back(heFaces.size());
  heFaces.push_back(he);

  he.src = vIdxList[1]-1;
  he.tgt = vIdxList[2]-1;
  fList.push_back(vIdxList[1] - 1);
  loopMap[vIdxList[0]-1].push_back(heFaces.size());
  heFaces.push_back(he);

  he.src = vIdxList[2]-1;
  he.tgt = vIdxList[0]-1;
  fList.push_back(vIdxList[2] - 1);
  loopMap[vIdxList[1]-1].push_back(heFaces.size());
  heFaces.push_back(he);

  return true;
}

bool PPM::objRead(const char *fName) {
  std::ifstream fStream(fName);
  if (!fStream.good())
    return false;

  // parse lines from the OBJ
  std::string line, type;
  while(!fStream.eof()) {
    std::getline(fStream, line);
    if (line.size() == 0)
      continue;

    std::stringstream ssLine(line);
    ssLine >> type;

    if (type == "v") {
      if (!objReadVtx(ssLine))
        fprintf(stderr, "vtx read err: %s\n", line.c_str());
    }

    if (type == "f") {
      if (!objReadFace(ssLine))
        fprintf(stderr, "face read err: %s\n", line.c_str());
    }
  }

  // check that vertex indices are valid
  nVtx = vList.size()/PPM_NVARS;
  for (const HeData &v : heFaces) {
    if (v.src < 0 || v.src >= nVtx || v.tgt < 0 || v.tgt >= nVtx) return false;
  }

  // get the vertex count
  nHe = heFaces.size();
  nFace = nHe / 3;

  return true;
}

void PPM::visInit() {
  glGenBuffers(1, &vboVtx); // vList.size() vertices (3 floats)
  glGenBuffers(1, &vboIdx); // fList.size() indices (1 int)
  glGenBuffers(1, &vboTessVtx); // vList.size() vertices (3 floats)
  glGenBuffers(1, &vboTessIdx); // fList.size() indices (1 int)

  fprintf(stderr, "binding base VAO\n");
  glGenVertexArrays(1, &vaoBase);
  glBindVertexArray(vaoBase);

  fprintf(stderr, "loading vidx vbo\n");
  glBindBuffer(GL_ARRAY_BUFFER, vboVtx);
  glBufferData(GL_ARRAY_BUFFER, PPM_NVARS * nVtx*sizeof(float), &vList[0], GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, PPM_NVARS * sizeof(float), (const void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, PPM_NVARS * sizeof(float), (const void*)(3* sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, PPM_NVARS * sizeof(float), (const void*)(6* sizeof(float)));

  fprintf(stderr, "loading fidx vbo\n");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * nFace*sizeof(int), &fList[0], GL_STATIC_DRAW);
  //glEnableVertexAttribArray(3);
  //glVertexAttribIPointer(3, GL_INT, 3, 0, (const void*)0);

  fprintf(stderr, "binding tess VAO\n");
  glGenVertexArrays(1, &vaoTess);
  glBindVertexArray(vaoTess);

  fprintf(stderr, "loading vtx tess vbo\n");
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glBufferData(GL_ARRAY_BUFFER, PPM_NVARS * (nFace*(nSub-1)*(nSub)/2 + nHe*(nSub-1)/2 + nVtx) * sizeof(float), 0, GL_STATIC_DRAW);
  cudaGraphicsGLRegisterBuffer(&dev_vboTessVtx, vboTessVtx, cudaGraphicsMapFlagsNone);
  checkCUDAError("cudaGraphicsGLRegisterBuffer", __LINE__);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, PPM_NVARS * sizeof(float), (const void*)0);
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, PPM_NVARS * sizeof(float), (const void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, PPM_NVARS * sizeof(float), (const void*)(6* sizeof(float)));

  fprintf(stderr, "loading fidx tess vbo\n");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboTessIdx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * nFace * nSubFace * sizeof(int), 0, GL_STATIC_DRAW);
  cudaGraphicsGLRegisterBuffer(&dev_vboTessIdx, vboTessIdx, cudaGraphicsMapFlagsNone);
  checkCUDAError("cudaGraphicsGLRegisterBuffer", __LINE__);

  glBindVertexArray(0);
}

void PPM::draw(Shader *vShader, Shader *tShader) {
  if (!isBuilt)
    return;
  
	if (visSkel) {
	  glPointSize(1.0f);
	  vShader->setUniform("uColor", 0.8f, 0.2f, 0.1f);
    vShader->setUniform("nShade", false);
    glBindVertexArray(vaoBase);
    vShader->bindIndexData(vboIdx);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, 0);
    
    glPointSize(3.0f);
    vShader->setUniform("uColor", 0.2f, 0.1f, 0.8f);
    vShader->setUniform("nShade", false);
    glBindVertexArray(vaoTess);
    vShader->bindIndexData(vboTessIdx);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, 3 * nFace*nSubFace, GL_UNSIGNED_INT, 0);
    glDrawElements(GL_POINTS, 3 * nFace*nSubFace, GL_UNSIGNED_INT, 0);
  }

  if (visFill) {
    glPointSize(3.0f);
    vShader->setUniform("uColor", 0.4f, 0.4f, 0.4f);
    vShader->setUniform("nShade", true);
    vShader->setUniform("dbgNormals", visDbgNormals);
    glBindVertexArray(vaoTess);
    vShader->bindIndexData(vboTessIdx);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, 3 * nFace*nSubFace, GL_UNSIGNED_INT, 0);
  }

  glBindVertexArray(0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


void PPM::visFree() {
  if (!useVisualize)
    return;

  glDeleteBuffers(1,&vboVtx); // vList.size() vertices (3 floats)
  glDeleteBuffers(1,&vboIdx); // vList.size() vertices (3 floats)
  cudaGraphicsUnregisterResource(dev_vboTessIdx);
  checkCUDAError("GraphicsUnregisterResource", __LINE__);
  cudaGraphicsUnregisterResource(dev_vboTessVtx);
  checkCUDAError("GraphicsUnregisterResource", __LINE__);
  glDeleteBuffers(1, &vboTessVtx); // vList.size() vertices (3 floats)
  glDeleteBuffers(1, &vboTessIdx); // vList.size() vertices (3 floats)
  glDeleteVertexArrays(1, &vaoBase); // vList.size() vertices (3 floats)
  glDeleteVertexArrays(1, &vaoTess); // vList.size() vertices (3 floats)
}

void PPM::cudaProbe() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0)
      throw std::runtime_error("No CUDA device detected");

    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      checkCUDAError("cudaGetDeviceProperties", __LINE__);
      
      fprintf(stderr, "Device Number: %d\n", i);
      fprintf(stderr, "  Device name: %s\n", prop.name);
      fprintf(stderr, "  Compute capability: %d.%d", prop.major, prop.minor);
      if (prop.major < 3) {
        fprintf(stderr, " (< 3.0, disabling texSamp)");
        canUseTexObjs = false;
      } else {
        canUseTexObjs = true;
      }
      fprintf(stderr, "\n");
    }
  }

