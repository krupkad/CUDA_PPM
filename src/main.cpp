#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <ctime>

#include <algorithm>

#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

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
    updateCamera();

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
    canvas = new PpmCanvas(ppm, shader, window);
    canvas->setBackgroundColor({ 0, 0, 0, 255 });
    canvas->setDrawBorder(false);
    canvas->setSize({ w - tools->width(), h });
    
   
    printf("PpmGui: performing layout\n");
    performLayout();
  }

  virtual ~PpmGui() {
    delete shader;
    delete ppm;
  }
  
  void rebuild() {
    if (fName.empty())
      return;
    canvas->setVisible(false);
    ppm->rebuild(fName.c_str(), nBasis, nSamp, nSub);
    canvas->setVisible(true);
    ppmBaseFaceBox->setValue(ppm->nFace);
  }

  void updateCamera() {
    //camPos.x = zoom * sin(phi) * sin(theta);
    //camPos.y = zoom * cos(theta);
    //camPos.z = -zoom * cos(phi) * sin(theta);
    camPos = glm::vec3(0.0f, 0.0f, -zoom);


    projection = glm::perspective(fovy, float(xSize) / float(ySize), zNear, zFar);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 model = glm::eulerAngleYX(phi, theta);
    projection = projection * view * model;

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
      ppmTimeBox->setValue(ppmTime / nbFrames);
      fpsTimeBox->setValue(float(nbFrames) / (currentTime - fpsTime));
      ppmTessRateBox->setValue(1000.0f * nbFrames * ppm->nFace * ppm->nSubFace / ppmTime);
      nbFrames = 0;
      ppmTime = 0.0f;
      fpsTime += 1.0;
    }
  }

  virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) {
    if (Screen::keyboardEvent(key, scancode, action, modifiers))
      return true;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      setVisible(false);
      glfwSetWindowShouldClose(glfwWindow(), GL_TRUE);
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
  nanogui::IntBox<int> *ppmBaseFaceBox;
  nanogui::FloatBox<float> *ppmTessRateBox;

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
      try {
        ppm->useTessSM = false;
        float dt1 = ppm->update();
        ppm->useTessSM = true;
        float dt2 = ppm->update();
        printf("%d -> %.3f %.3f ms\n", nSamp, dt1, dt2);
      } catch (const std::exception &e) {
        printf("%d -> %s\n", nSamp, e.what());
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
      try {
        ppm->useTessSM = false;
        float dt1 = ppm->update();
        ppm->useTessSM = true;
        float dt2 = ppm->update();
        printf("%d -> %.3f %.3f ms\n", nBasis, dt1, dt2);
      } catch (const std::exception &e) {
        printf("%d -> %s\n", nSamp, e.what());
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
      try {
        ppm->useTessSM = false;
        float dt1 = ppm->update();
        ppm->useTessSM = true;
        float dt2 = ppm->update();
        printf("%d -> %.3f %.3f ms\n", nSub, dt1, dt2);
      } catch (const std::exception &e) {
        printf("%d -> %s\n", nSamp, e.what());
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
  nanogui::init();
  PpmGui *gui = new PpmGui(1200, 800);

  // main loop
  gui->setVisible(true);
  while (!glfwWindowShouldClose(gui->glfwWindow())) {
    glfwPollEvents();
    gui->drawAll();
  }

  delete gui;
  nanogui::shutdown();

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
