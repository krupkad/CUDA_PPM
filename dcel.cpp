#include "dcel.hpp"
#include "shader.hpp"

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include "glm/gtc/type_ptr.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>

DCEL::~DCEL() {
  devFree();
  delete vboDataBuf;
  glDeleteBuffers(1, &vboData);
  glDeleteBuffers(1, &vboIdx);
}

DCEL::DCEL(const char *fName) {
  objRead(fName);
  printf("read done\n");
  devInit();
  printf("dev done\n");
  visInit();
  printf("vis done\n");
}

bool DCEL::objReadVtx(std::istream &fStream) {
  glm::vec3 p;
  for (int i = 0; i < 3; i++)
    fStream >> p[i];

  vList.push_back(p);
  nrmList.push_back(glm::vec3());
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

  // no degenerate faces
  if (N < 3) {
    printf("degenerate face\n");
    return false;
  }

  // for a general polygon, triangulate via a center point
  if (N > 3) {
    glm::vec3 avg;
    for (int vIdx : vIdxList)
      avg += vList[vIdx];
    avg /= vIdxList.size();
    vList.push_back(avg);

    for (int i = 0; i < N; i++) {
      int j = (i+1)%N;

      heSrcList.push_back(vIdxList[j]-1);
      heDstList.push_back(vList.size()-1);

      heSrcList.push_back(vList.size()-1);
      heDstList.push_back(vIdxList[i]-1);

      //fList.push_back(vIdxList[i]-1);
      //fList.push_back(vIdxList[j]-1);
      //fList.push_back(vList.size()-1);
    }
  } else {
    fList.push_back(vIdxList[0]-1);
    fList.push_back(vIdxList[1]-1);
    fList.push_back(vIdxList[2]-1);
  }


  // add the side edges
  for (int i = 0; i < N; i++) {
    int j = (i+1)%N;
    heSrcList.push_back(vIdxList[i]-1);
    heDstList.push_back(vIdxList[j]-1);
  }

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
  for (int v : heSrcList) {
    if (v < 0 || v >= N)
      throw std::logic_error("DCEL: invalid vertex");
  }
  for (int v : heDstList) {
    if (v < 0 || v >= N)
      throw std::logic_error("DCEL: invalid vertex");
  }
}

void DCEL::visInit() {
  glGenBuffers(1, &vboData);
  glGenBuffers(1, &vboIdx);

  int stride = 3+3;
  ssoVtx = SHADER_SSO(3,stride,0);
  ssoNrm = SHADER_SSO(3,stride,3);

  vboDataBuf = new float[stride*vList.size()];
  for(int i = 0; i < vList.size(); i++) {
    memcpy(&vboDataBuf[stride*i], glm::value_ptr(vList[i]), 3*sizeof(float));
    memcpy(&vboDataBuf[stride*i+3], glm::value_ptr(nrmList[i]), 3*sizeof(float));
  }

  // Bind+upload vertex coordinates
  int szData = stride*vList.size()*sizeof(float);
  glBindBuffer(GL_ARRAY_BUFFER, vboData);
  glBufferData(GL_ARRAY_BUFFER, szData, vboDataBuf, GL_STATIC_DRAW);

  // Bind+upload the indices to the GL_ELEMENT_ARRAY_BUFFER.
  int szIdx = fList.size()*sizeof(int);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, szIdx, &fList[0], GL_STATIC_DRAW);
}

void DCEL::visDraw(Shader *shader) {
  shader->bindVertexData("Position", vboData, ssoVtx);
  //shader->bindVertexData("Normal", vboData, ssoNrm);
  shader->bindIndexData(vboIdx);
  glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
  glDrawElements(GL_TRIANGLES, fList.size(), GL_UNSIGNED_INT, 0);
}



