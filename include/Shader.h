#pragma once

#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map> // 追加

class Shader {
public:
    GLuint ID;

    Shader(const char* vertexPath, const char* fragmentPath);
    void use();
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setTexture(const std::string &name, int unit, GLuint texture); // const を削除

private:
    void checkCompileErrors(GLuint shader, std::string type);
    // uniformロケーションをキャッシュするためのマップを追加
    mutable std::unordered_map<std::string, GLint> uniformLocationCache;
    // uniformロケーションを取得するヘルパー関数を追加 (mutableラムダやconst_castを避けるため、setTextureからconstを削除)
    GLint getUniformLocation(const std::string &name);
};
