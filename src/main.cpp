#include <cstring>

#include "ppm.hpp"
#include "gui.hpp"

int main(int argc, char *argv[]) {
  if (argc > 1) {
    
    PPM *ppm = new PPM(false);
    
    int nBasis, nSamp, nSub;
    int r2 = atoi(argv[3]), r3 = atoi(argv[4]);
    if (!strcmp(argv[2], "--samp")) {
      if (argc != 7) {
        printf("usage: %s --samp [start end] deg sub\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      nBasis = atoi(argv[5]);
      nSub = atoi(argv[6]);
      for (int i = r2; i <= r3; i++) {
        nSamp = nSub * (1 << i);
        ppm->rebuild(argv[1], nBasis, nSamp, nSub);

        printf("%d -> ", nSamp);
        try {
          ppm->useTessSM = false;
          float dt = ppm->update();
          printf("%f ", dt);
        } catch (const std::exception &e) {
          printf("%s ", e.what());
        }
        try {
          ppm->useTessSM = true;
          float dt = ppm->update();
          printf("%f\n", dt);
        } catch (const std::exception &e) {
          printf("%s\n", e.what());
        }
      }
    }
    
    else if (!strcmp(argv[2], "--deg")) {
      if (argc != 6) {
        printf("usage: %s --deg [start end] sub\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      nSub = atoi(argv[5]);
      for (int i = r2; i <= r3; i++) {
        nSamp = nBasis = 2*(2 + i);
        ppm->rebuild(argv[1], nBasis, nSamp, nSub);
        
        printf("%d -> ", nBasis);
        try {
          ppm->useTessSM = false;
          float dt = ppm->update();
          printf("%f ", dt);
        } catch (const std::exception &e) {
          printf("%s ", e.what());
        }
        try {
          ppm->useTessSM = true;
          float dt = ppm->update();
          printf("%f\n", dt);
        } catch (const std::exception &e) {
          printf("%s\n", e.what());
        }

      }
    }
    
    else if (!strcmp(argv[2], "--sub")) {
      if (argc != 7) {
        printf("usage: %s --sub [start end] deg samp\n", argv[0]);
        delete ppm;
        return 0;
      }
      
      nBasis = atoi(argv[5]);
      nSamp = atoi(argv[6]);
      for (int i = r2; i <= r3; i++) {
        nSub = (1 << i);
        ppm->rebuild(argv[1], nBasis, nSamp, nSub);
        
        printf("%d -> ", nSub);
        try {
          ppm->useTessSM = false;
          float dt = ppm->update();
          printf("%f ", dt);
        } catch (const std::exception &e) {
          printf("%s ", e.what());
        }
        try {
          ppm->useTessSM = true;
          float dt = ppm->update();
          printf("%f\n", dt);
        } catch (const std::exception &e) {
          printf("%s\n", e.what());
        }
      }
    }
    
    else {
      printf("usage: %s [--samp, --deg, --sub]\n", argv[0]);
    }
    
    delete ppm;
    return 0;
  }
  
  // initialize graphics
  PpmGui gui(1200, 800);
  gui.mainLoop();

  return 0;
}

/*
void resizeCallback(GLFWwindow* window, int x, int y) {
  xSize = x;
  ySize = y;
  glViewport(0,0,x,y);
  updateCamera();
}

void errorCallback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (leftMousePressed) {
    phi += (xpos - lastX) / xSize;
    theta -= (ypos - lastY) / ySize;
    updateCamera();
  }

  lastX = xpos;
  lastY = ypos;
}

void scrollCallback(GLFWwindow* window, double dx, double dy) {
  zoom += 30.0f*dy / ySize;
  zoom = std::fmax(0.01f, std::fmin(zoom, 100.0f));
  updateCamera();
}
*/
