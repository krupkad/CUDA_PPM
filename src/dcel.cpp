#include "dcel.hpp"
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


DCEL::DCEL(const char *fName, bool glVis) :
  useTessSM(false),
  canUseTexObjs(true),
  useSampTex(true),
  useSvdUpdate(true),
  useVisualize(glVis),
  visFill(true),
  visSkel(true),
  visDbgNormals(true),
  inFile(fName)
{}

DCEL::~DCEL() {
  devFree();
  visFree();
}

void DCEL::rebuild(int nBasis, int nGrid, int nSub) {
  objRead(inFile.c_str());
  printf("read done\n");

  // tesselation controls
  this->nSub = nSub;
  this->nSubFace = nSub*nSub;
  this->nSubVtx = (nSub + 1)*(nSub + 2) / 2;

  if (useVisualize) {
    visInit();
    printf("vis done\n");
  }

  this->nBasis = nBasis;
  this->nGrid = nGrid;
  this->nBasis2 = nBasis*nBasis;
  this->nGrid2 = nGrid*nGrid;
  devInit();
  printf("dev done\n");
}

// sort half edges a) by source, b) in patch boundary order
void DCEL::getHeLoops() {
  for(auto &itr : loopMap) {
    std::vector<int> &loop = itr.second;

    // check if there's a vertex with no predecessor i.e.
    // we're on a boundary, and make it first
    int first;
    for (first = 0; first < loop.size(); first++) {
      bool hasPred = false;
      for (int i = first+1; i < loop.size(); i++) {
        if (heFaces[loop[first]].x == heFaces[loop[i]].y) {
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
      int dst = heFaces[loop[i]].y;
      for (int j = i+1; j < loop.size(); j++) {
        int src = heFaces[loop[j]].x;
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
    for (int heIdx : loopMap[v1])
      heLoops.push_back(make_int4(v1, heFaces[heIdx].x, 0, 0));
	  vRange.y = heLoops.size();
    vBndList.push_back(vRange);

    if (vRange.y - vRange.x < degMin)
      degMin = vRange.y - vRange.x;
    if (vRange.y - vRange.x > degMax)
      degMax = vRange.y - vRange.x;
  }
  nDeg = degMax - degMin + 1;
}

bool DCEL::objReadVtx(std::istream &fStream) {
  float p;
  for (int i = 0; i < 3; i++) {
    fStream >> p;
    vList.push_back(p);
  }
  for (int i = 0; i < 5; i++)
    vList.push_back(0.0f);

  return true;
}

bool DCEL::objReadFace(std::istream &fStream) {
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
      vIdxList.push_back(vList.size()/8 + vIdx + 1);
  }
  unsigned int N = vIdxList.size();

  if (N != 3) {
    printf("nontriangular face\n");
    return false;
  }

  int4 he;

  he = make_int4(vIdxList[0]-1, vIdxList[1]-1, 0, 0);
  fList.push_back(vIdxList[0] - 1);
  loopMap[vIdxList[2]-1].push_back(heFaces.size());
  heFaces.push_back(he);

  he = make_int4(vIdxList[1] - 1, vIdxList[2] - 1, 0, 0);
  fList.push_back(vIdxList[1] - 1);
  loopMap[vIdxList[0]-1].push_back(heFaces.size());
  heFaces.push_back(he);

  he = make_int4(vIdxList[2] - 1, vIdxList[0] - 1, 0, 0);
  fList.push_back(vIdxList[2] - 1);
  loopMap[vIdxList[1]-1].push_back(heFaces.size());
  heFaces.push_back(he);

  return true;
}

void DCEL::objRead(const char *fName) {
  std::ifstream fStream(fName);

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
        printf("vtx read err: %s\n", line.c_str());
    }

    if (type == "f") {
      if (!objReadFace(ssLine))
        printf("face read err: %s\n", line.c_str());
    }
  }

  // check that vertex indices are valid
  unsigned int N = vList.size()/8;
  for (const int4 &v : heFaces) {
    if (v.x < 0 || v.x >= N || v.y < 0 || v.y >= N)
      throw std::logic_error("DCEL: invalid vertex");
  }

  // get the vertex count
  nVtx = vList.size()/8;
  nHe = heFaces.size();
  nFace = nHe / 3;
}

void DCEL::visInit() {
  glGenBuffers(1, &vboVtx); // vList.size() vertices (3 floats)
  glGenBuffers(1, &vboIdx); // fList.size() indices (1 int)
  glGenBuffers(1, &vboTessVtx); // vList.size() vertices (3 floats)
  glGenBuffers(1, &vboTessIdx); // fList.size() indices (1 int)

  printf("binding base VAO\n");
  glGenVertexArrays(1, &vaoBase);
  glBindVertexArray(vaoBase);

  printf("loading vidx vbo\n");
  glBindBuffer(GL_ARRAY_BUFFER, vboVtx);
  glBufferData(GL_ARRAY_BUFFER, 8 * nVtx*sizeof(float), &vList[0], GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)(3* sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)(6* sizeof(float)));

  printf("loading fidx vbo\n");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * nFace*sizeof(int), &fList[0], GL_STATIC_DRAW);
  //glEnableVertexAttribArray(3);
  //glVertexAttribIPointer(3, GL_INT, 3, 0, (const void*)0);

  printf("binding tess VAO\n");
  glGenVertexArrays(1, &vaoTess);
  glBindVertexArray(vaoTess);

  printf("loading vtx tess vbo\n");
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glBufferData(GL_ARRAY_BUFFER, 8 * nFace * nSubVtx * sizeof(float), 0, GL_STATIC_DRAW);
  cudaGraphicsGLRegisterBuffer(&dev_vboTessVtx, vboTessVtx, cudaGraphicsMapFlagsNone);
  checkCUDAError("cudaGraphicsGLRegisterBuffer", __LINE__);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)0);
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, vboTessVtx);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)(6* sizeof(float)));

  printf("loading fidx tess vbo\n");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboTessIdx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * nFace * nSubFace * sizeof(int), 0, GL_STATIC_DRAW);
  cudaGraphicsGLRegisterBuffer(&dev_vboTessIdx, vboTessIdx, cudaGraphicsMapFlagsNone);
  checkCUDAError("cudaGraphicsGLRegisterBuffer", __LINE__);

  glBindVertexArray(0);
}

void DCEL::draw(Shader *vShader, Shader *tShader) {
	if (visSkel) {
	  glPointSize(1.0f);
	  glLineWidth(2.0f);
	  vShader->setUniform("uColor", 0.8f, 0.2f, 0.1f);
    vShader->setUniform("nShade", false);
    glBindVertexArray(vaoBase);
    //vShader->bindVertexData("Position", vboVtx, SHADER_SSO(3, 8, 0));
    //vShader->bindVertexData("Normal", vboVtx, SHADER_SSO(3, 8, 3));
    //vShader->bindVertexData("UV", vboVtx, SHADER_SSO(2, 8, 6));
    vShader->bindIndexData(vboIdx);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, 0);

    glPointSize(3.0f);
    glLineWidth(1.0f);
    vShader->setUniform("uColor", 0.3, 0.3f, 0.0f);
    vShader->setUniform("nShade", false);
    glBindVertexArray(vaoTess);
    //vShader->bindVertexData("Position", vboTessVtx, SHADER_SSO(3, 8, 0));
    //vShader->bindVertexData("Normal", vboTessVtx, SHADER_SSO(3, 8, 3));
    //vShader->bindVertexData("UV", vboTessVtx, SHADER_SSO(2, 8, 6));
    vShader->bindIndexData(vboTessIdx);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, 3 * nFace*nSubFace, GL_UNSIGNED_INT, 0);
    glDrawElements(GL_POINTS, 3 * nFace*nSubFace, GL_UNSIGNED_INT, 0);
	}

  if (visFill) {
    glPointSize(3.0f);
    glLineWidth(1.0f);
    vShader->setUniform("uColor", 0.4f, 0.4f, 0.4f);
    vShader->setUniform("nShade", true);
    vShader->setUniform("dbgNormals", visDbgNormals);
    glBindVertexArray(vaoTess);
    //vShader->bindVertexData("Position", vboTessVtx, SHADER_SSO(3, 8, 0));
    //vShader->bindVertexData("Normal", vboTessVtx, SHADER_SSO(3, 8, 3));
    //vShader->bindVertexData("UV", vboTessVtx, SHADER_SSO(2, 8, 6));
    vShader->bindIndexData(vboTessIdx);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, 3 * nFace*nSubFace, GL_UNSIGNED_INT, 0);
  }

  glBindVertexArray(0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


void DCEL::visFree() {
  if (!useVisualize)
    return;

  glDeleteBuffers(1,&vboVtx); // vList.size() vertices (3 floats)
  glDeleteBuffers(1,&vboIdx); // vList.size() vertices (3 floats)
  cudaGraphicsUnregisterResource(dev_vboTessIdx);
  cudaGraphicsUnregisterResource(dev_vboTessVtx);
  glDeleteBuffers(1, &vboTessVtx); // vList.size() vertices (3 floats)
  glDeleteBuffers(1, &vboTessIdx); // vList.size() vertices (3 floats)
}



