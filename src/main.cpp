#include <cstring>

#include "ppm.hpp"
#include "gui.hpp"

int main(int argc, char *argv[]) {
  printf("argc = %d\n", argc);
  if (argc > 2) {
    
    PPM *ppm = new PPM(false);
    const char *fName = argv[1];
    const char *opt = argv[2];
    
    int nBasis, nSamp, nSub;
    if (!strcmp(opt, "--samp")) {
      if (argc != 7) {
        printf("usage: %s --samp start end deg sub\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      int r2 = atoi(argv[3]), r3 = atoi(argv[4]);
      nBasis = atoi(argv[5]);
      nSub = atoi(argv[6]);
      for (int i = r2; i <= r3; i++) {
        nSamp = nSub * (1 << i);
        ppm->rebuild(fName, nBasis, nSamp, nSub);

        printf("%d -> ", nSamp);
        try {
          ppm->useTessSM = false;
          float dt = ppm->update(-1, glm::vec3(0.0f),.01f);
          printf("%f ", dt);
        } catch (const std::exception &e) {
          printf("%s ", e.what());
        }
        try {
          ppm->useTessSM = true;
          float dt = ppm->update(-1, glm::vec3(0.0f),.01f);
          printf("%f\n", dt);
        } catch (const std::exception &e) {
          printf("%s\n", e.what());
        }
      }
    }
    
    else if (!strcmp(opt, "--deg")) {
      if (argc != 6) {
        printf("usage: %s --deg start end sub\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      int r2 = atoi(argv[3]), r3 = atoi(argv[4]);
      nSub = atoi(argv[5]);
      for (int i = r2; i <= r3; i++) {
        nSamp = nBasis = 2*(2 + i);
        ppm->rebuild(fName, nBasis, nSamp, nSub);
        
        printf("%d -> ", nBasis);
        try {
          ppm->useTessSM = false;
          float dt = ppm->update(-1, glm::vec3(0.0f),.01f);
          printf("%f ", dt);
        } catch (const std::exception &e) {
          printf("%s ", e.what());
        }
        try {
          ppm->useTessSM = true;
          float dt = ppm->update(-1, glm::vec3(0.0f),.01f);
          printf("%f\n", dt);
        } catch (const std::exception &e) {
          printf("%s\n", e.what());
        }

      }
    }
    
    else if (!strcmp(opt, "--sub")) {
      if (argc != 7) {
        printf("usage: %s --sub start end deg samp\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      int r2 = atoi(argv[3]), r3 = atoi(argv[4]);
      nBasis = atoi(argv[5]);
      nSamp = atoi(argv[6]);
      for (int i = r2; i <= r3; i++) {
        nSub = (1 << i);
        ppm->rebuild(fName, nBasis, nSamp, nSub);
        
        printf("%d -> ", nSub);
        try {
          ppm->useTessSM = false;
          float dt = ppm->update(-1, glm::vec3(0.0f),.01f);
          printf("%f ", dt);
        } catch (const std::exception &e) {
          printf("%s ", e.what());
        }
        try {
          ppm->useTessSM = true;
          float dt = ppm->update(-1, glm::vec3(0.0f),.01f);
          printf("%f\n", dt);
        } catch (const std::exception &e) {
          printf("%s\n", e.what());
        }
      }
    }

    else if (!strcmp(opt, "--ix")) {
      if (argc != 5) {
        printf("usage: %s --ix sub count\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      nBasis = 4;
      nSamp = 4;
      nSub = (1 << atoi(argv[3]));
      ppm->rebuild(fName, nBasis, nSamp, nSub);
      ppm->update(-1, glm::vec3(0.0f),.1f);

      float dt = 0, t;
      float2 uv;
      srand(time(NULL));
      for (int i = 0; i < atoi(argv[4]); i++) {
        float phi = 2.0f*float(rand())*M_PI/RAND_MAX, theta = float(rand())*M_PI/RAND_MAX;
        glm::vec3 p0(cos(phi)*cos(theta), sin(phi), cos(phi)*sin(theta));
        float t0 = clock();
        ppm->intersect(p0, -p0);
        float t1 = clock();
        dt += (t1-t0)/CLOCKS_PER_SEC;
      }
      printf("ix mean = %.3f us\n", 1.0e6f*dt/atoi(argv[4]));
    }

    else {
      printf("usage: %s [--samp, --deg, --sub, --ix]\n", argv[0]);
    }
    
    delete ppm;
    return 0;
  }
  
  // initialize graphics
  nanogui::init();
  PpmGui gui(1200, 800);
  gui.mainLoop();

  return 0;
}

