#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <ctime>

#include <algorithm>

#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include <nanogui/nanogui.h>

#include <cuda_runtime.h>

#include "ppm.hpp"
#include "shader.hpp"
#include "util/error.hpp"


class PpmCanvas : public nanogui::GLCanvas {
public:
  PpmCanvas(PPM *ppm, Shader *shader, nanogui::Widget *parent) :
    nanogui::GLCanvas(parent),
    shader(shader),
    ppm(ppm)
  {}

  virtual void drawGL() override {
    int oldBlendSrc, oldBlendDst;
    glGetIntegerv(GL_BLEND_SRC_RGB, &oldBlendSrc);
    glGetIntegerv(GL_BLEND_DST_RGB, &oldBlendDst);
    glBlendFunc(GL_ONE, GL_ZERO);
    glEnable(GL_DEPTH_TEST);
    ppm->draw(shader, nullptr);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(oldBlendSrc, oldBlendDst);
  }

private:
  Shader *shader;
  PPM *ppm;
};

int cudaProbe(PPM *ppm) {
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
      ppm->canUseTexObjs = false;
    } else {
      ppm->canUseTexObjs = true;
    }
    printf("\n");
  }
	return nDevices;
}

class PpmGui : public nanogui::Screen {
public:
  PpmGui(int w, int h) :
    nanogui::Screen(nanogui::Vector2i(w, h), "PPM Demo"),
    ppm(nullptr),
    xSize(w), ySize(h),
    leftMousePressed(false), rightMousePressed(false),
    fovy(M_PI / 4), zNear(0.1), zFar(100.0),
    theta(1.22), phi(-0.7), zoom(5.0),
    ppmTime(0.0), fpsTime(0.0), nbFrames(0),
    fName(""), nBasis(4), nSamp(4), nSub(2)
  {
    using namespace nanogui;

    printf("PpmGui: creating window\n");
    Widget *window = new Window(this, "");
    window->setPosition(Vector2i(0,0));
    Layout *wLayout = new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5);
    window->setLayout(wLayout); 

    printf("PpmGui: initialize glew\n");
    glewExperimental = GL_TRUE;
    glewInit();

    printf("PpmGui: creating ppm\n");
    ppm = new PPM(true);

    printf("PpmGui: compiling shader\n");
    shader = new Shader();
    shader->setShader("shaders/ppm.frag.glsl", GL_FRAGMENT_SHADER);
    shader->setShader("shaders/ppm.vert.glsl", GL_VERTEX_SHADER);
    shader->recompile();
    glClearDepth(1.0);
    glDepthFunc(GL_LEQUAL);
    updateCamera();

    printf("PpmGui: creating UI\n");
    tools = new Widget(window);
    tools->setLayout(new GroupLayout(5));

    CheckBox *c0 = new CheckBox(tools, "Show Wireframe");
    c0->setChecked(ppm->visSkel);
    c0->setCallback([this](bool value) { ppm->visSkel = value; });

    CheckBox *c1 = new CheckBox(tools, "Show Normals");
    c1->setChecked(ppm->visDbgNormals);
    c1->setCallback([this](bool value) { ppm->visDbgNormals = value; });

    CheckBox *c2 = new CheckBox(tools, "Show Surface");
    c2->setChecked(ppm->visFill);
    c2->setCallback([this](bool value) { ppm->visFill = value; });
    
    new Label(tools, "nBasis");
    IntBox<int> *c3 = new IntBox<int>(tools, nBasis);
    new Label(tools, "nSamp");
    IntBox<int> *c4 = new IntBox<int>(tools, nSamp);
    new Label(tools, "nSub");
    IntBox<int> *c5 = new IntBox<int>(tools, nSub);
    
    c3->setMinValue(3);
    c3->setCallback([this,c4](int value) { 
      nBasis = value;
      nSamp = std::max(nBasis, nSamp);
      c4->setValue(nSamp);
      c4->setMinValue(nBasis);
      canvas->setVisible(false);
      ppm->rebuild(fName.c_str(), nBasis, nSamp, nSub);
      canvas->setVisible(true);
    });
    c3->setEditable(true);
    c3->setSpinnable(true);
    
    c4->setMinValue(nBasis);
    c4->setCallback([this](int value) { 
      nSamp = value;
      canvas->setVisible(false);
      ppm->rebuild(fName.c_str(), nBasis, nSamp, nSub);
      canvas->setVisible(true);
    });
    c4->setEditable(true);
    c4->setSpinnable(true);
    
    c5->setMinValue(1);
    c5->setCallback([this](int value) { 
      nSub = value;
      canvas->setVisible(false);
      ppm->rebuild(fName.c_str(), nBasis, nSamp, nSub);
      canvas->setVisible(true);
    });
    c5->setEditable(true);
    c5->setSpinnable(true);
    
    //double(nbFrames) / (currentTime - fpsTime), ppmTime / nbFrames
    new Label(tools, "PPM time (ms)");
    ppmTimeBox = new FloatBox<float>(tools);
    new Label(tools, "Frame Rate (fps)");
    fpsTimeBox = new FloatBox<float>(tools);
    
    performLayout(); // to calculate toolbar width

    printf("PpmGui: creating canvas\n");
    canvas = new PpmCanvas(ppm, shader, window);
    canvas->setBackgroundColor({ 0, 0, 0, 255 });
    canvas->setDrawBorder(false);
    canvas->setSize({ w - tools->width(), h });
    
    // check CUDA devices
    int nDevices = cudaProbe(ppm);
    if (!nDevices) {
      printf("no CUDA device found\n");
      throw;
    }

    printf("PpmGui: performing layout\n");
    performLayout();
  }

  virtual ~PpmGui() {
    delete shader;
    delete ppm;
  }
  
  void rebuild(const char *f) {
    fName = f;
    ppm->rebuild(f, nBasis, nSamp, nSub);
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
  
  virtual bool resizeEvent(const nanogui::Vector2i &size) {
    if (Screen::resizeEvent(size))
      return true;
    
    performLayout();  // to recalculate toolbar width
    canvas->setSize({ size[0] - tools->width(), size[1] });
    xSize = size[0];
    ySize = size[1];
    performLayout();
    updateCamera();
    
    return true;
  }

  virtual void draw(NVGcontext *ctx) {
    ppmTime += ppm->update();
    Screen::draw(ctx);

    // perform timing
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - fpsTime >= 1.0) {
      printf("%.1f fps (dt = %.3g ms)\n", double(nbFrames) / (currentTime - fpsTime), ppmTime / nbFrames);
      ppmTimeBox->setValue(ppmTime / nbFrames);
      fpsTimeBox->setValue(float(nbFrames) / (currentTime - fpsTime));
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
      glfwSetWindowShouldClose(glfwWindow(), GL_TRUE);
      return true;
    }

    //// visualization
    // F = show PPM surface
    // S = show input skeleton
    // N = color using normals
    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
      ppm->visFill = !ppm->visFill;
      return true;
    }
    if (key == GLFW_KEY_S && action == GLFW_PRESS) {
      ppm->visSkel = !ppm->visSkel;
      return true;
    }
    if (key == GLFW_KEY_N && action == GLFW_PRESS) {
      ppm->visDbgNormals = !ppm->visDbgNormals;
      return true;
    }

    //// functional controls
    // 1 = SM use for vertex computation
    // 2 = keep evaluated bezier values in textures
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
      ppm->useTessSM = !ppm->useTessSM;
      if (ppm->useTessSM) {
        printf("using full-SM tessVtx\n");
      }
      else {
        printf("using non-SM tessVtx\n");
      }
      return true;
    }

    if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
      ppm->useSampTex = !ppm->useSampTex;
      if (ppm->useSampTex)
        printf("using texture sampling patterns\n");
      else
        printf("using computed sampling patterns\n");
      return true;
    }

    // D = toggle deformation
    if (key == GLFW_KEY_D && action == GLFW_PRESS) {
      ppm->useSvdUpdate = !ppm->useSvdUpdate;
      if (ppm->useSvdUpdate)
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
    
    if (key == GLFW_KEY_LEFT_BRACKET && action == GLFW_PRESS) {
      zoom *= 1.1;
      updateCamera();
      return true;
    }
    if (key == GLFW_KEY_RIGHT_BRACKET && action == GLFW_PRESS) {
      zoom /= 1.1;
      updateCamera();
      return true;
    }

    return false;
  }

private:
  PPM *ppm;
  std::string fName;
  int nBasis, nSamp, nSub;

  nanogui::GLCanvas *canvas;
  nanogui::Widget *tools;
  
  int xSize, ySize;
  float lastX, lastY, xpos, ypos;
  bool leftMousePressed, rightMousePressed;
  float fovy, zNear, zFar;
  float theta, phi, zoom;
  glm::vec3 camPos;
  glm::mat4 projection;

  float ppmTime, fpsTime;
  int nbFrames;
  nanogui::FloatBox<float> *ppmTimeBox, *fpsTimeBox;

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
  PpmGui *gui = new PpmGui(1200, 800);

  // main loop
  gui->rebuild(argv[4]);
  gui->setVisible(true);
  while (!glfwWindowShouldClose(gui->glfwWindow())) {
    glfwPollEvents();
    gui->drawAll();
  }

  nanogui::shutdown();
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
