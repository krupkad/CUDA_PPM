#ifndef SHADER_H
#define SHADER_H

//#include <map>
#include <unordered_map>
#include <string>
#include <iostream>

#include <GL/glew.h>
#include <GL/gl.h>
#include "glm/glm.hpp"

#define SHADER_SSO(size,stride,offset) \
  (((size)&0x1f) | (((stride)&0x1f) << 5) | (((offset)&0x1f) << 10))
#define SHADER_SSO_SIZE(sso) ((sso)&0x1f)
#define SHADER_SSO_STRIDE(sso) (((sso) >> 5)&0x1f)
#define SHADER_SSO_OFFSET(sso) (((sso) >> 10)&0x1f)

void printGLErrorLog();

class Shader {
  public:
    Shader();
    Shader(std::string src);
    virtual ~Shader();

    void requireAttr(std::string name);
    void requireAttr(std::string name, int loc);
    void requireUniform(std::string name);
    void requireUniform(std::string name, int loc);

    GLuint setShader(const std::string &src, GLenum shType);

    void setUniform(std::string name, const glm::vec3& v);
    void setUniform(std::string name, float x, float y, float z);
    void setUniform(std::string name, bool v);
    void setUniform(std::string name, int v);
    void setUniform(std::string name, const glm::mat4& m);

    void bindVertexData(std::string name, unsigned int vbo);
    void bindVertexData(std::string name, unsigned int vbo, int sso);
    void bindIndexData(unsigned int vbo);
    void unbindVertexData(std::string name);

    void recompile();

  protected:

    int resolveUniform(std::string name);
    int resolveAttribute(std::string name);

    void use();

    std::unordered_map<GLuint,GLuint> shMap;
    GLuint program;
    bool hasProgram;
    std::unordered_map<std::string,unsigned int> uMap;
    std::unordered_map<std::string,unsigned int> aMap;

    static Shader *inUse;
};

#endif /* SHADER_H */
