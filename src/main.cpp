#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <ctime>

#include <algorithm>

#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include <nanogui/nanogui.h>

#include <cuda_runtime.h>

#include "dcel.hpp"
#include "shader.hpp"
#include "util/error.hpp"


class PpmCanvas : public nanogui::GLCanvas {
public:
  PpmCanvas(DCEL *dcel, Shader *shader, nanogui::Widget *parent) :
    nanogui::GLCanvas(parent),
    shader(shader),
    dcel(dcel)
  {}

  virtual void drawGL() override {
    int oldBlendSrc, oldBlendDst;
    glGetIntegerv(GL_BLEND_SRC_RGB, &oldBlendSrc);
    glGetIntegerv(GL_BLEND_DST_RGB, &oldBlendDst);
    glBlendFunc(GL_ONE, GL_ZERO);
    glEnable(GL_DEPTH_TEST);
    dcel->draw(shader, nullptr);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(oldBlendSrc, oldBlendDst);
  }

private:
  Shader *shader;
  DCEL *dcel;
};

int cudaProbe(DCEL *dcel) {
	int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    checkCUDAError("cudaGetDeviceProperties", __LINE__);
    
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d", prop.major, prop.minor);
    if (prop.major < 3) {
      printf(" (< 3.0, disabling texSamp)");
      dcel->canUseTexObjs = false;
    } else {
      dcel->canUseTexObjs = true;
    }
    printf("\n");
  }
	return nDevices;
}

class PpmGui : public nanogui::Screen {
public:
  DCEL *dcel;

  PpmGui(const char *src, int w, int h) :
    nanogui::Screen(nanogui::Vector2i(w, h), (const char *)"PPM Demo"),
    xSize(w), ySize(h),
    leftMousePressed(false), rightMousePressed(false),
    fovy(M_PI / 4), zNear(0.1), zFar(100.0),
    theta(1.22), phi(-0.7), zoom(5.0),
    ppmTime(0.0), fpsTime(0.0), nbFrames(0)
  {
    using namespace nanogui;

    printf("PpmGui: creating window\n");
    Widget *window = new Window(this);
    window->setPosition(Vector2i(0,0));
    window->setLayout(new GroupLayout()); 

    printf("PpmGui: initialize glew\n");
    glewExperimental = GL_TRUE;
    glewInit();

    printf("PpmGui: creating dcel\n");
    dcel = new DCEL(src, true);

    printf("PpmGui: compiling shader\n");
    shader = new Shader();
    shader->setShader("shaders/dcel.frag.glsl", GL_FRAGMENT_SHADER);
    shader->setShader("shaders/dcel.vert.glsl", GL_VERTEX_SHADER);
    shader->recompile();
    glClearDepth(1.0);
    glDepthFunc(GL_LEQUAL);
    updateCamera();

    printf("PpmGui: creating canvas\n");
    canvas = new PpmCanvas(dcel, shader, window);
    canvas->setBackgroundColor({ 0, 0, 0, 255 });
    canvas->setDrawBorder(false);
    canvas->setSize({ w - 25, h  - 75 });

    printf("PpmGui: creating UI\n");
    Widget *tools = new Widget(window);
    tools->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5));

    CheckBox *c0 = new CheckBox(tools, "Show Wireframe");
    c0->setChecked(dcel->visSkel);
    c0->setCallback([this](bool value) { dcel->visSkel = value; });

    CheckBox *c1 = new CheckBox(tools, "Show Normals");
    c1->setChecked(dcel->visDbgNormals);
    c1->setCallback([this](bool value) { dcel->visDbgNormals = value; });

    CheckBox *c2 = new CheckBox(tools, "Show Surface");
    c2->setChecked(dcel->visFill);
    c2->setCallback([this](bool value) { dcel->visFill = value; });

    printf("PpmGui: performing layout\n");
    performLayout();

    //printGLErrorLog();
  }

  virtual ~PpmGui() {
    delete shader;
  }

  void updateCamera() {
    theta = std::min((float)M_PI - 0.001f, std::max(0.001f, theta));

    camPos.x = zoom * sin(phi) * sin(theta);
    camPos.y = zoom * cos(theta);
    camPos.z = -zoom * cos(phi) * sin(theta);

    projection = glm::perspective(fovy, float(xSize) / float(ySize), zNear, zFar);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    projection = projection * view;

    shader->setUniform("model", projection);
    shader->setUniform("invTrModel", glm::inverse(glm::transpose(projection)));
    shader->setUniform("CamDir", glm::normalize(-camPos));
  }

  virtual void draw(NVGcontext *ctx) {
    ppmTime += dcel->update();
    Screen::draw(ctx);

    // perform timing
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - fpsTime >= 1.0) {
      printf("%.1f fps (dt = %.3g ms)\n", double(nbFrames) / (currentTime - fpsTime), ppmTime / nbFrames);
      nbFrames = 0;
      ppmTime = 0.0f;
      fpsTime += 1.0;
    }
  }

  virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) {
    if (Screen::keyboardEvent(key, scancode, action, modifiers))
      return true;

    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
      setVisible(false);
      return true;
    }

    //// visualization
    // F = show PPM surface
    // S = show input skeleton
    // N = color using normals
    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
      dcel->visFill = !dcel->visFill;
      return true;
    }
    if (key == GLFW_KEY_S && action == GLFW_PRESS) {
      dcel->visSkel = !dcel->visSkel;
      return true;
    }
    if (key == GLFW_KEY_N && action == GLFW_PRESS) {
      dcel->visDbgNormals = !dcel->visDbgNormals;
      return true;
    }

    //// functional controls
    // 1 = SM use for vertex computation
    // 2 = keep evaluated bezier values in textures
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
      dcel->useTessSM = !dcel->useTessSM;
      if (dcel->useTessSM) {
        printf("using full-SM tessVtx\n");
      }
      else {
        printf("using non-SM tessVtx\n");
      }
      return true;
    }

    if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
      dcel->useSampTex = !dcel->useSampTex;
      if (dcel->useSampTex)
        printf("using texture sampling patterns\n");
      else
        printf("using computed sampling patterns\n");
      return true;
    }

    // D = toggle deformation
    if (key == GLFW_KEY_D && action == GLFW_PRESS) {
      dcel->useSvdUpdate = !dcel->useSvdUpdate;
      if (dcel->useSvdUpdate)
        printf("using deformation\n");
      else
        printf("using static\n");
      return true;
    }

    // I,J,K,L = camera controls
    if (key == GLFW_KEY_J && action == GLFW_PRESS) {
      phi += M_PI / 18.0;
      updateCamera();
      return true;
    }
    if (key == GLFW_KEY_L && action == GLFW_PRESS) {
      phi -= M_PI / 18.0;
      updateCamera();
      return true;
    }
    if (key == GLFW_KEY_K && action == GLFW_PRESS) {
      theta += M_PI / 18.0;
      updateCamera();
      return true;
    }
    if (key == GLFW_KEY_I && action == GLFW_PRESS) {
      theta -= M_PI / 18.0;
      updateCamera();
      return true;
    }
    
    if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS) {
      zoom *= 1.1;
      updateCamera();
      return true;
    }
    if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS) {
      zoom /= 1.1;
      updateCamera();
      return true;
    }

    return false;
  }

private:
  nanogui::GLCanvas *canvas;
  
  int xSize, ySize;
  float lastX, lastY, xpos, ypos;
  bool leftMousePressed, rightMousePressed;
  float fovy, zNear, zFar;
  float theta, phi, zoom;
  glm::vec3 camPos;
  glm::mat4 projection;

  float ppmTime, fpsTime;
  int nbFrames;

  Shader *shader;
};

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("usage: %s nBasis nSamp nSub objFile\n", argv[0]);
    return 0;
  }
  int nBasis = atoi(argv[1]);
  int nSamp = atoi(argv[2]);
  int nSub = atoi(argv[3]);

  // initialize graphics
  nanogui::init();
  PpmGui *gui = new PpmGui(argv[4], 1200, 800);

  // check CUDA devices
  int nDevices = cudaProbe(gui->dcel);
  if (!nDevices) {
    printf("no CUDA device found\n");
    delete gui;
    return 0;
  }

  // build the PPM
  gui->dcel->rebuild(nBasis, nSamp, nSub);
  printf("created dcel\n");
  //printGLErrorLog();

  gui->setVisible(true);
  while (!glfwWindowShouldClose(gui->glfwWindow()))
    gui->drawAll();

  delete gui;

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
