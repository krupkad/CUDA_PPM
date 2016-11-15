#include "dcel.hpp"
#include "shader.hpp"

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "glm/gtc/type_ptr.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <algorithm>


DCEL::DCEL(const char *fName) {
  degMin = INT_MAX;
  degMax = INT_MIN;

  objRead(fName);
  printf("read done\n");
  heSort();
  printf("loop sort done\n");
  devInit();
  printf("dev done\n");
  visInit();
  printf("vis done\n");

  cudaDeviceSynchronize();
}

DCEL::~DCEL() {
  devFree();
  visFree();
}

// sort half edges a) by source, b) in patch boundary order
void DCEL::heSort() {
  for(auto &itr : loopMap) {
    std::vector<int> &loop = itr.second;

    // check if there's a vertex with no predecessor i.e.
    // we're on a boundary, and make it first
    int first;
    for (first = 0; first < loop.size(); first++) {
      bool hasPred = false;
      for (int i = first+1; i < loop.size(); i++) {
        if (heList[loop[first]].x == heList[loop[i]].y) {
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
      int dst = heList[loop[i]].y;
      for (int j = i+1; j < loop.size(); j++) {
        int src = heList[loop[j]].x;
        if (src == dst) {
          std::swap(loop[i+1],loop[j]);
          break;
        }
      }
    }
  }

  // create a halfedge list ordered by both source vertex and loop sequence
  std::vector<int4> new_heList;
  for (int v1 = 0; v1 < vList.size(); v1++) {
    int2 vRange;
    vRange.x = new_heList.size();
    for (int heIdx : loopMap[v1])
      new_heList.push_back(make_int4(v1, heList[heIdx].x, heList[heIdx].y, heList[heIdx].w));
    vRange.y = new_heList.size();
    vBndList.push_back(vRange);
    if (vRange.y - vRange.x < degMin)
      degMin = vRange.y - vRange.x;
    if (vRange.y - vRange.x > degMax)
      degMax = vRange.y - vRange.x;
  }
  heList = new_heList;
}

bool DCEL::objReadVtx(std::istream &fStream) {
  glm::vec3 p;
  for (int i = 0; i < 3; i++)
    fStream >> p[i];
  vList.push_back(p);

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

    vIdxList.push_back(vIdx);
  }
  unsigned int N = vIdxList.size();

  if (N != 3) {
    printf("nontriangular face\n");
    return false;
  }

  glm::ivec3 tri;
  int fCnt = fList.size() / 3;
  for (int i = 0; i < 3; i++) {
    tri[i] = vIdxList[i] - 1;
    fList.push_back(tri[i]);
  }
  triList.push_back(tri);

  int4 he;

  he = make_int4(vIdxList[0]-1, vIdxList[1]-1, vIdxList[2]-1, 3*fCnt + 0);
  loopMap[vIdxList[2]-1].push_back(heList.size());
  heList.push_back(he);

  he = make_int4(vIdxList[1]-1, vIdxList[2]-1, vIdxList[0]-1, 3*fCnt + 1);
  loopMap[vIdxList[0]-1].push_back(heList.size());
  heList.push_back(he);

  he = make_int4(vIdxList[2]-1, vIdxList[0]-1, vIdxList[1]-1, 3*fCnt + 2);
  loopMap[vIdxList[1]-1].push_back(heList.size());
  heList.push_back(he);

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
  unsigned int N = vList.size();
  for (const int4 &v : heList) {
    if (v.x < 0 || v.x >= N || v.y < 0 || v.y >= N)
      throw std::logic_error("DCEL: invalid vertex");
  }
}

void DCEL::visInit() {
  glGenBuffers(1, &vboVtxList); // vList.size() vertices (3 floats)
  glGenBuffers(1, &vboIdx); // fList.size() indices (1 int)
  //cudaGraphicsGLRegisterBuffer(&cuVtxResource, vboData, cudaGraphicsMapFlagsNone);

  int vCnt = vList.size();
  int fIdxCnt = fList.size();

  printf("loading vtx vbo\n");
  vboVtxListBuf = new float[3*vCnt];
  for(int i = 0; i < vCnt; i++)
    memcpy(&vboVtxListBuf[3*i], glm::value_ptr(vList[i]), 3*sizeof(float));
  glBindBuffer(GL_ARRAY_BUFFER, vboVtxList);
  glBufferData(GL_ARRAY_BUFFER, 3*vCnt*sizeof(float), vboVtxListBuf, GL_STATIC_DRAW);

  printf("loading fidx vbo\n");
  glBindBuffer(GL_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ARRAY_BUFFER, fIdxCnt*sizeof(int), &fList[0], GL_STATIC_DRAW);
}

void DCEL::visDraw(Shader *vShader, Shader *tShader) {
  glPointSize(3.0f);

  glBindBuffer(GL_ARRAY_BUFFER, vboVtxList);
  glBufferData(GL_ARRAY_BUFFER, 3*nVtx*sizeof(float), vboVtxListBuf, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ARRAY_BUFFER, 3*nFace*sizeof(int), &fList[0], GL_STATIC_DRAW);

  glLineWidth(3.0f);
  vShader->setUniform("uColor", 1.0f, 0.0f, 0.0f);
  vShader->bindVertexData("Position", vboVtxList, SHADER_SSO(3,3,0));
  vShader->bindIndexData(vboIdx);
  glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
  glDrawElements(GL_TRIANGLES, 3*nFace, GL_UNSIGNED_INT, 0);

  glBindBuffer(GL_ARRAY_BUFFER, vboVtxList);
  glBufferData(GL_ARRAY_BUFFER, 3*nFace*nSubVtx*sizeof(float), tessVtx, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ARRAY_BUFFER, 3*nFace*nSubFace*sizeof(int), tessIdx, GL_STATIC_DRAW);

  glLineWidth(1.0f);
  vShader->setUniform("uColor", 0.0f, 1.0f, 0.0f);
  vShader->bindVertexData("Position", vboVtxList, SHADER_SSO(3,3,0));
  vShader->bindIndexData(vboIdx);
  glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
  glDrawElements(GL_TRIANGLES, 3*nFace*nSubFace, GL_UNSIGNED_INT, 0);
  glDrawElements(GL_POINTS, 3*nFace*nSubFace, GL_UNSIGNED_INT, 0);
}


void DCEL::visFree() {
  delete vboVtxListBuf;
  glDeleteBuffers(1,&vboVtxList); // vList.size() vertices (3 floats)
  glDeleteBuffers(1,&vboIdx); // vList.size() vertices (3 floats)
  cudaGraphicsUnregisterResource(cuVtxResource);
}



