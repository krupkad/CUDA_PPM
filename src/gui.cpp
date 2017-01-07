#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <ctime>

#include <algorithm>

#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <cuda_runtime.h>

#include "gui.hpp"
#include "ppm.hpp"
#include "util/error.hpp"


class PpmCanvas : public nanogui::GLCanvas {
  public:
    PpmCanvas(PPM *ppm, Shader *shader, PpmGui *gui, nanogui::Widget *parent) :
      nanogui::GLCanvas(parent),
      shader(shader),
      ppm(ppm),
      gui(gui)
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
    
    virtual bool mouseButtonEvent(const nanogui::Vector2i &pos, int button, bool down, int mods) override {
      if (down) {
        gui->click(pos[0] - position()[0], pos[1]- position()[1]);
      }
      return true;
    }

  private:
    Shader *shader;
    PPM *ppm;
    PpmGui* gui;
};

PpmGui::PpmGui(int w, int h) :
    nanogui::Screen(nanogui::Vector2i(w, h), "PPM Demo"),
    ppm(nullptr),
    xSize(w), ySize(h),
    leftMousePressed(false), rightMousePressed(false),
    fovy(M_PI / 4), zNear(0.1), zFar(100.0),
    yAngle(0), xAngle(0), zoom(5.0),
    ppmTime(0.0), fpsTime(0.0), nbFrames(0),
    fName(""), nBasis(4), nSamp(4), nSub(2)
{
  using namespace nanogui;

  printf("PpmGui: creating window\n");
  Widget *window = new Window(this, "");
  window->setPosition(Vector2i(0,0));
  window->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5)); 

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

  printf("PpmGui: creating UI\n");
  tools = new Widget(window);
  tools->setLayout(new GroupLayout(8));

  CheckBox *chkWireframe = new CheckBox(tools, "Show Wireframe");
  chkWireframe->setChecked(ppm->visSkel);
  chkWireframe->setCallback([this](bool value) { ppm->visSkel = value; });

  CheckBox *chkNormals = new CheckBox(tools, "Show Normals");
  chkNormals->setChecked(ppm->visDbgNormals);
  chkNormals->setCallback([this](bool value) { ppm->visDbgNormals = value; });

  CheckBox *chkFill = new CheckBox(tools, "Show Surface");
  chkFill->setChecked(ppm->visFill);
  chkFill->setCallback([this](bool value) { ppm->visFill = value; });
  
  CheckBox *chkSM = new CheckBox(tools, "Use SM");
  chkSM->setChecked(ppm->useTessSM);
  chkSM->setCallback([this](bool value) { ppm->useTessSM = value; });

  if (ppm->canUseTexObjs) {
    CheckBox *chkSampTex = new CheckBox(tools, "Use Tex Samp");
    chkSampTex->setChecked(ppm->useSampTex);
    chkSampTex->setCallback([this](bool value) { ppm->useSampTex = value; });
  }
  
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
    rebuild();
  });
  c3->setEditable(true);
  c3->setSpinnable(true);
  
  c4->setMinValue(nBasis);
  c4->setCallback([this](int value) { 
    nSamp = value;
    rebuild();
  });
  c4->setEditable(true);
  c4->setSpinnable(true);
  
  c5->setMinValue(1);
  c5->setCallback([this](int value) { 
    nSub = value;
    rebuild();
  });
  c5->setEditable(true);
  c5->setSpinnable(true);
  
  new Label(tools, "PPM time (ms)");
  ppmTimeBox = new FloatBox<float>(tools);
  new Label(tools, "Frame Rate (fps)");
  fpsTimeBox = new FloatBox<float>(tools);
  
  new Label(tools, "Base faces");
  ppmBaseFaceBox = new IntBox<int>(tools);
  new Label(tools, "Tess rate (faces/s)");
  ppmTessRateBox = new FloatBox<float>(tools);
  
  Button *fileButton = new Button(tools, "Load OBJ");
  fileButton->setCallback([this,fileButton] { 
      fName = nanogui::file_dialog({{"obj","Wavefront OBJ"}}, true);
      rebuild();
      fileButton->setTooltip(fName);
  });


  performLayout(); // to calculate toolbar width

  printf("PpmGui: creating canvas\n");
  canvas = new PpmCanvas(ppm, shader, this, window);
  canvas->setBackgroundColor({ 0, 0, 0, 255 });
  canvas->setDrawBorder(false);
  canvas->setSize({ w - tools->width(), h });
  
 
  printf("PpmGui: performing layout\n");
  performLayout();

  updateCamera();
}

PpmGui::~PpmGui() {
  delete shader;
  delete ppm;
  nanogui::shutdown();
}

void PpmGui::mainLoop() {
  // main loop
  setVisible(true);
  while (!glfwWindowShouldClose(glfwWindow())) {
    glfwPollEvents();
    drawAll();
  }
}

void PpmGui::click(int x, int y) {
  float ratio = float(canvas->width())/canvas->height();
  float fovx = atan(tan(fovy) * ratio);
  glm::vec3 dz = glm::normalize(-camPos);
  glm::vec3 dx = glm::cross(dz, up);
  glm::vec3 dy = glm::cross(dx, dz);


  float vx = float(x) / canvas->width();
  float vy = float(y) / canvas->height();

        printf("click %f %f\n", vx, vy);
  glm::vec3 dir = glm::normalize(dz + tanf(fovx/2)*(2.0f*vx-1)*dx + tanf(fovy/2)*(1.0f - 2.0f*vy)*dy);
  float2 uv;
  ppm->intersect(camPos, dir, uv);
}

void PpmGui::rebuild() {
  if (fName.empty())
    return;
  canvas->setVisible(false);
  ppm->rebuild(fName.c_str(), nBasis, nSamp, nSub);
  canvas->setVisible(true);
  ppmBaseFaceBox->setValue(ppm->nFace);
}

void PpmGui::updateCamera() {
  camPos.x = zoom * cos(yAngle) * sin(xAngle);
  camPos.y = zoom * sin(yAngle);
  camPos.z = zoom * cos(yAngle) * cos(xAngle);

  right = glm::vec3(sin(xAngle - M_PI/2), 0, cos(xAngle - M_PI/2));
  up = glm::normalize(glm::cross(right, -camPos));


  glm::mat4 projection = glm::perspective(fovy, float(canvas->width()) / float(canvas->height()), zNear, zFar);
  glm::mat4 view = glm::lookAt(camPos, glm::vec3(0,0,0), up);
  mvp = projection*view;

  shader->setUniform("model", projection*view);
  shader->setUniform("invTrModel", glm::inverse(glm::transpose(projection*view)));
  shader->setUniform("CamDir", glm::normalize(-camPos));
}

bool PpmGui::resizeEvent(const nanogui::Vector2i &size) {
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

void PpmGui::draw(NVGcontext *ctx) {
  ppmTime += ppm->update();
  Screen::draw(ctx);

  // perform timing
  double currentTime = glfwGetTime();
  nbFrames++;
  if (currentTime - fpsTime >= 1.0) {
    ppmTimeBox->setValue(ppmTime / nbFrames);
    fpsTimeBox->setValue(float(nbFrames) / (currentTime - fpsTime));
    ppmTessRateBox->setValue(1000.0f * nbFrames * ppm->nFace * ppm->nSubFace / ppmTime);
    nbFrames = 0;
    ppmTime = 0.0f;
    fpsTime += 1.0;
  }
}

bool PpmGui::keyboardEvent(int key, int scancode, int action, int modifiers) {
  if (Screen::keyboardEvent(key, scancode, action, modifiers))
    return true;

  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    setVisible(false);
    glfwSetWindowShouldClose(glfwWindow(), GL_TRUE);
    return true;
  }

  // I,J,K,L = camera controls
  if (key == GLFW_KEY_J && action == GLFW_PRESS) {
    xAngle += M_PI / 18.0;
    updateCamera();
    return true;
  }
  if (key == GLFW_KEY_L && action == GLFW_PRESS) {
    xAngle -= M_PI / 18.0;
    updateCamera();
    return true;
  }
  if (key == GLFW_KEY_K && action == GLFW_PRESS) {
    yAngle += M_PI / 18.0;
    updateCamera();
    return true;
  }
  if (key == GLFW_KEY_I && action == GLFW_PRESS) {
    yAngle -= M_PI / 18.0;
    updateCamera();
    return true;
  }
  
  if (key == GLFW_KEY_LEFT_BRACKET && action == GLFW_PRESS) {
    fovy *= 1.1;
    updateCamera();
    return true;
  }
  if (key == GLFW_KEY_RIGHT_BRACKET && action == GLFW_PRESS) {
    fovy /= 1.1;
    updateCamera();
    return true;
  }

  return false;
}


