#include <render/shaders/Shaders.hpp>
#include <helpers/cm/ColorManagement.hpp>
#include <hyprutils/memory/Casts.hpp>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <format>
#include <sstream>
#include <stdexcept>

using namespace NColorManagement;
using namespace Hyprutils::Memory;

static std::string shaderSource(std::string_view name, bool mirror, bool tonemap = false, eTransferFunction sourceTF = CM_TRANSFER_FUNCTION_SRGB,
                                eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR) {
    if (name == "defines.h")
        return std::format("#define USE_CM 1\n#define USE_RGBA 1\n#define USE_MIRROR {}\n#define USE_TONEMAP {}\n"
                           "#define USE_BLUR 0\n#define USE_BLUR_MATTE 0\n#define USE_BLUR_ALPHA_MASK 0\n"
                           "#define USE_DISCARD 0\n#define USE_TINT 0\n#define USE_ROUNDING 0\n"
                           "#define USE_MOTION_BLUR 0\n#define USE_SDR_MOD 0\n#define USE_ICC 0\n#define USE_ALT_TONEMAP 0\n"
                           "#define SOURCE_TF {}\n#define TARGET_TF {}\n",
                           mirror ? 1 : 0, tonemap ? 1 : 0, sc<int>(sourceTF), sc<int>(targetTF));

    const auto it = std::ranges::find(SHADERS, name, &std::pair<std::string_view, std::string_view>::first);
    if (it == SHADERS.end())
        throw std::runtime_error(std::format("Missing shader {}", name));

    std::istringstream input{std::string{it->second}};
    std::string        source, line;
    while (std::getline(input, line)) {
        if (line.starts_with("#include \""))
            source += shaderSource(std::string_view{line}.substr(10, line.size() - 11), mirror, tonemap, sourceTF, targetTF);
        else if (!line.starts_with("#extension GL_ARB_shading_language_include"))
            source += line + '\n';
    }
    return source;
}

class CColorManagementMirrorTest : public testing::Test {
  protected:
    void SetUp() override;
    void TearDown() override;
    void createProgram(bool mirror, bool tonemap = false, std::string_view fragment = "surface.frag", eTransferFunction sourceTF = CM_TRANSFER_FUNCTION_SRGB,
                       eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR);
    void checkPixel(eTransferFunction tf, float reference, float sourceMax, float encoded, float expected, float alpha = 1.0f, bool tonemap = false);

    // Offscreen GL resources.
    EGLDisplay            m_display  = EGL_NO_DISPLAY;
    EGLContext            m_context  = EGL_NO_CONTEXT;
    EGLSurface            m_surface  = EGL_NO_SURFACE;
    GLuint                m_program  = 0;
    GLuint                m_fbo      = 0;
    std::array<GLuint, 3> m_textures = {};
};

void CColorManagementMirrorTest::SetUp() {
    m_display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (m_display == EGL_NO_DISPLAY || !eglInitialize(m_display, nullptr, nullptr))
        GTEST_SKIP() << "Surfaceless EGL is unavailable";

    ASSERT_TRUE(eglBindAPI(EGL_OPENGL_ES_API));
    const EGLint configAttrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
    EGLConfig    config        = nullptr;
    EGLint       count         = 0;
    ASSERT_TRUE(eglChooseConfig(m_display, configAttrs, &config, 1, &count));
    if (count == 0)
        GTEST_SKIP() << "OpenGL ES 3 is unavailable";

    const EGLint contextAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    m_context                   = eglCreateContext(m_display, config, EGL_NO_CONTEXT, contextAttrs);
    ASSERT_NE(m_context, EGL_NO_CONTEXT);
    const EGLint surfaceAttrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    m_surface                   = eglCreatePbufferSurface(m_display, config, surfaceAttrs);
    ASSERT_NE(m_surface, EGL_NO_SURFACE);
    ASSERT_TRUE(eglMakeCurrent(m_display, m_surface, m_surface, m_context));

    const std::string extensions = rc<const char*>(glGetString(GL_EXTENSIONS));
    if (!extensions.contains("GL_EXT_color_buffer_float"))
        GTEST_SKIP() << "Floating-point framebuffer attachments are unavailable";

    glGenTextures(m_textures.size(), m_textures.data());
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    for (size_t i = 0; i < m_textures.size(); ++i) {
        glBindTexture(GL_TEXTURE_2D, m_textures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, nullptr);
        if (i > 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i - 1, GL_TEXTURE_2D, m_textures[i], 0);
    }
    const GLenum attachments[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, attachments);
    ASSERT_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 1, 1);
    glDisable(GL_DITHER);
}

void CColorManagementMirrorTest::TearDown() {
    if (m_context != EGL_NO_CONTEXT && eglGetCurrentContext() == m_context) {
        glDeleteProgram(m_program);
        glDeleteFramebuffers(1, &m_fbo);
        glDeleteTextures(m_textures.size(), m_textures.data());
    }
    if (m_display == EGL_NO_DISPLAY)
        return;
    eglMakeCurrent(m_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (m_surface != EGL_NO_SURFACE)
        eglDestroySurface(m_display, m_surface);
    if (m_context != EGL_NO_CONTEXT)
        eglDestroyContext(m_display, m_context);
    eglTerminate(m_display);
}

void CColorManagementMirrorTest::createProgram(bool mirror, bool tonemap, std::string_view fragment, eTransferFunction sourceTF, eTransferFunction targetTF) {
    glDeleteProgram(m_program);
    m_program                                                   = glCreateProgram();
    const std::array<std::pair<GLenum, std::string>, 2> sources = {{
        {GL_VERTEX_SHADER,
         "#version 300 es\nout vec2 v_texcoord;\nvoid main() { v_texcoord = vec2(0.5); "
         "vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n"},
        {GL_FRAGMENT_SHADER, shaderSource(fragment, mirror, tonemap, sourceTF, targetTF)},
    }};
    for (const auto& [type, source] : sources) {
        const auto  shader = glCreateShader(type);
        const auto* text   = source.c_str();
        glShaderSource(shader, 1, &text, nullptr);
        glCompileShader(shader);
        GLint                  compiled = GL_FALSE;
        std::array<char, 4096> log      = {};
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        glGetShaderInfoLog(shader, log.size(), nullptr, log.data());
        glAttachShader(m_program, shader);
        glDeleteShader(shader);
        ASSERT_EQ(compiled, GL_TRUE) << log.data();
    }
    glLinkProgram(m_program);
    GLint                  linked = GL_FALSE;
    std::array<char, 4096> log    = {};
    glGetProgramiv(m_program, GL_LINK_STATUS, &linked);
    glGetProgramInfoLog(m_program, log.size(), nullptr, log.data());
    ASSERT_EQ(linked, GL_TRUE) << log.data();
    glUseProgram(m_program);
}

void CColorManagementMirrorTest::checkPixel(eTransferFunction tf, float reference, float sourceMax, float encoded, float expected, float alpha, bool tonemap) {
    SCOPED_TRACE(std::format("tf={}, reference={}, encoded={}, alpha={}", sc<int>(tf), reference, encoded, alpha));
    std::array<float, 4> monitorWithMirror = {};
    for (bool mirror : {true, false}) {
        ASSERT_NO_FATAL_FAILURE(createProgram(mirror, tonemap, "surface.frag", tf));
        const std::array<float, 4> pixel = {encoded * alpha, encoded * alpha, encoded * alpha, alpha};
        glBindTexture(GL_TEXTURE_2D, m_textures[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel.data());
        glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
        glUniform1f(glGetUniformLocation(m_program, "alpha"), 1.0f);
        glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), reference);
        glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, sourceMax);
        glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), 0.0f, 10000.0f);
        const std::array<float, 9> identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        glUniformMatrix3fv(glGetUniformLocation(m_program, "convertMatrix"), 1, GL_FALSE, identity.data());
        glUniformMatrix3fv(glGetUniformLocation(m_program, "targetPrimariesXYZ"), 1, GL_FALSE, identity.data());
        glUniform1f(glGetUniformLocation(m_program, "maxLuminance"), 1000.0f);
        glUniform1f(glGetUniformLocation(m_program, "dstMaxLuminance"), 518.0f);
        glUniform1f(glGetUniformLocation(m_program, "dstRefLuminance"), 203.0f);
        glUniform1i(glGetUniformLocation(m_program, "tonemapMode"), 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        std::array<float, 4> monitor = {};
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, monitor.data());
        if (!mirror) {
            for (size_t i = 0; i < monitor.size(); ++i)
                EXPECT_NEAR(monitor[i], monitorWithMirror[i], 0.0001f);
            continue;
        }
        monitorWithMirror            = monitor;
        std::array<float, 4> capture = {};
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, capture.data());
        for (size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(capture[i], expected * alpha, 0.002f);
        EXPECT_NEAR(capture[3], alpha, 0.0001f);
        EXPECT_EQ(glGetError(), GL_NO_ERROR);
    }
}

TEST_F(CColorManagementMirrorTest, PQReferenceWhiteAndMidtones) {
    for (float reference : {80.0f, 203.0f, 308.0f, 500.0f}) {
        for (float srgb : {0.0f, 0.125f, 0.25f, 0.5f, 0.75f, 1.0f}) {
            const float linear = srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
            const float scaled = std::pow(linear * reference / 10000.0f, 0.1593017578125f);
            const float pq     = std::pow((0.8359375f + 18.8515625f * scaled) / (1.0f + 18.6875f * scaled), 78.84375f);
            checkPixel(CM_TRANSFER_FUNCTION_ST2084_PQ, reference, 10000.0f, pq, srgb);
        }
    }
}

TEST_F(CColorManagementMirrorTest, LinearHDRReferenceWhiteAndTransparency) {
    for (float alpha : {0.0f, 0.25f, 0.5f, 1.0f})
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 80.0f, 308.0f / 80.0f, 1.0f, alpha);
}

TEST_F(CColorManagementMirrorTest, HLGReferenceWhite) {
    const float hlg = 0.17883277f * std::log(12.0f * 0.308f - 0.28466892f) + 0.55991073f;
    checkPixel(CM_TRANSFER_FUNCTION_HLG, 308.0f, 1000.0f, hlg, 1.0f);
}

TEST_F(CColorManagementMirrorTest, SDRIgnoresHDRReferenceWhite) {
    checkPixel(CM_TRANSFER_FUNCTION_SRGB, 308.0f, 308.0f, 0.5f, 0.5f);
    checkPixel(CM_TRANSFER_FUNCTION_GAMMA22, 308.0f, 308.0f, std::pow(0.21404114f, 1.0f / 2.2f), 0.5f);
}

TEST_F(CColorManagementMirrorTest, MissingReferenceWhiteIsFinite) {
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 0.0f, 80.0f, 1.0f, 1.0f);
}

TEST_F(CColorManagementMirrorTest, TonemappingVariantsCompile) {
    for (auto sourceTF : {CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_ST2084_PQ, CM_TRANSFER_FUNCTION_HLG, CM_TRANSFER_FUNCTION_EXT_LINEAR}) {
        for (auto targetTF : {CM_TRANSFER_FUNCTION_LINEAR, CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_ST2084_PQ}) {
            SCOPED_TRACE(std::format("sourceTF={}, targetTF={}", sc<int>(sourceTF), sc<int>(targetTF)));
            for (auto fragment : {"surface.frag", "border.frag"}) {
                ASSERT_NO_FATAL_FAILURE(createProgram(true, false, fragment, sourceTF, targetTF));
                ASSERT_NO_FATAL_FAILURE(createProgram(true, true, fragment, sourceTF, targetTF));
                ASSERT_NO_FATAL_FAILURE(createProgram(false, true, fragment, sourceTF, targetTF));
            }
            ASSERT_NO_FATAL_FAILURE(createProgram(false, false, "blurfinish.frag", sourceTF, targetTF));
            ASSERT_NO_FATAL_FAILURE(createProgram(false, false, "blurprepare.frag", sourceTF, targetTF));
        }
    }
}

TEST_F(CColorManagementMirrorTest, ExtendedSRGBReferenceWhite) {
    const float encoded = 1.055f * std::pow(308.0f / 80.0f, 1.0f / 2.4f) - 0.055f;
    checkPixel(CM_TRANSFER_FUNCTION_EXT_SRGB, 308.0f, 80.0f, encoded, 1.0f);
}

TEST_F(CColorManagementMirrorTest, MonitorTonemappingDoesNotChangeCapture) {
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 80.0f, 308.0f / 80.0f, 1.0f, 1.0f, true);
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 80.0f, 308.0f * 0.21404114f / 80.0f, 0.5f, 1.0f, true);
}

TEST_F(CColorManagementMirrorTest, ParametricLinearWhiteUsesDeclaredEncodingRange) {
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 308.0f, 1.0f, 1.0f);
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 1000.0f, 0.308f, 1.0f);
}
