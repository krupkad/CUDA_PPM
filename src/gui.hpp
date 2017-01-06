#ifndef PPM_GUI_H
#define PPM_GUI_H

#include <glm/glm.hpp>
#include <nanogui/nanogui.h>

#include "ppm.hpp"
#include "shader.hpp"

class PpmGui : public nanogui::Screen {
  public:
    PpmGui(int w, int h);
    virtual ~PpmGui();
    void mainLoop();
    void click(int x, int y);
    void rebuild();
    void updateCamera();
    virtual bool resizeEvent(const nanogui::Vector2i &size);
    virtual void draw(NVGcontext *ctx);
    virtual bool keyboardEvent(int key, int scancode, int action, int modifiers);
  
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

#endif /* PPM_GUI_H */

