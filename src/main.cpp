#include <cstring>
#include <cmath>
#include <vector>

#include "ppm.hpp"
#include "gui.hpp"

int main(int argc, char *argv[]) {

  // initialize graphics
  try {
    nanogui::init();
    PpmGui gui(1200, 800);
    gui.mainLoop();
  } catch (const std::exception &e) {
    printf("%s\n", e.what());
  }

  return 0;
}

