#include "dcel.hpp"
#include <sstream>
#include <stdexcept>

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
    if (sscanf(vDesc.c_str(), "%d", &vIdx) < tokLen)
      return false;

    vIdxList.push_back(vIdx);
  }
  unsigned int N = vIdxList.size();

  // no degenerate faces
  if (N < 3)
    return false;

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
    }
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("missing arg\n");
    return 0;
  }

  DCEL dcel(argv[1]);
  return 0;
}
