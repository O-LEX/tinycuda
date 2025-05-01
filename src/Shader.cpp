#include "Shader.h"
#include <iostream> // エラー出力用にインクルードを確認

Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    // シェーダーのコードを読み込む
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;

    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        // ファイルを開く
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;

        // ファイルのバッファ内容をストリームに読み込む
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        
        // ストリームの内容を文字列に変換する
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
        
        // ファイルを閉じる
        vShaderFile.close();
        fShaderFile.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl; // 詳細なエラーメッセージ
        ID = 0; // エラー発生フラグとしてIDを0に設定
        return; // コンストラクタを終了
    }

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    // シェーダーをコンパイルする
    GLuint vertex, fragment;
    
    // 頂点シェーダー
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, nullptr);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");
    
    // フラグメントシェーダー
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, nullptr);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");

    // シェーダープログラムをリンクする
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    // シェーダーを削除する
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use() {
    glUseProgram(ID);
}

// uniformロケーションを取得するヘルパー関数
GLint Shader::getUniformLocation(const std::string &name) {
    // キャッシュを確認
    if (uniformLocationCache.find(name) != uniformLocationCache.end()) {
        return uniformLocationCache[name];
    }
    // キャッシュにない場合は取得してキャッシュする
    GLint location = glGetUniformLocation(ID, name.c_str());
    if (location == -1) {
        // Uniformが見つからない場合のエラーメッセージ（デバッグに役立つ）
        // 本番環境ではコメントアウトするか、より洗練されたエラー処理を行う
        // std::cerr << "Warning: uniform '" << name << "' not found in shader program " << ID << std::endl;
    }
    uniformLocationCache[name] = location;
    return location;
}

void Shader::setBool(const std::string &name, bool value) const {
    // getUniformLocationを使用するように変更 (constメンバー関数内でキャッシュを変更するため、キャッシュマップをmutableにする)
    GLint location = const_cast<Shader*>(this)->getUniformLocation(name);
    if (location != -1) {
        glUniform1i(location, (int)value);
    }
}

void Shader::setInt(const std::string &name, int value) const {
    // getUniformLocationを使用するように変更
    GLint location = const_cast<Shader*>(this)->getUniformLocation(name);
     if (location != -1) {
        glUniform1i(location, value);
    }
}

void Shader::setFloat(const std::string &name, float value) const {
    // getUniformLocationを使用するように変更
    GLint location = const_cast<Shader*>(this)->getUniformLocation(name);
     if (location != -1) {
        glUniform1f(location, value);
    }
}

// setTextureはキャッシュを変更する可能性があるため、constを削除
void Shader::setTexture(const std::string &name, int unit, GLuint texture) {
    // getUniformLocationを使用するように変更
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(location, unit);
    }
}

void Shader::checkCompileErrors(GLuint shader, std::string type) {
    int success;
    char infoLog[1024];
    if (type == "PROGRAM") {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    } else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}
