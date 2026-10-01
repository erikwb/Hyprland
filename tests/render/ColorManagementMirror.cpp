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
                                eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR, bool altTonemap = false) {
    if (name == "defines.h")
        return std::format("#define USE_CM 1\n#define USE_RGBA 1\n#define USE_MIRROR {}\n#define USE_TONEMAP {}\n"
                           "#define USE_BLUR 0\n#define USE_BLUR_MATTE 0\n#define USE_BLUR_ALPHA_MASK 0\n"
                           "#define USE_DISCARD 0\n#define USE_TINT 0\n#define USE_ROUNDING 0\n"
                           "#define USE_MOTION_BLUR 0\n#define USE_SDR_MOD 0\n#define USE_ICC 0\n#define USE_ALT_TONEMAP {}\n"
                           "#define SOURCE_TF {}\n#define TARGET_TF {}\n",
                           mirror ? 1 : 0, tonemap ? 1 : 0, altTonemap ? 1 : 0, sc<int>(sourceTF), sc<int>(targetTF));

    const auto IT = std::ranges::find(SHADERS, name, &std::pair<std::string_view, std::string_view>::first);
    if (IT == SHADERS.end())
        throw std::runtime_error(std::format("Missing shader {}", name));

    std::istringstream input{std::string{IT->second}};
    std::string        source, line;
    while (std::getline(input, line)) {
        if (line.starts_with("#include \""))
            source += shaderSource(std::string_view{line}.substr(10, line.size() - 11), mirror, tonemap, sourceTF, targetTF, altTonemap);
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
                       eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR, bool altTonemap = false);

    // Linear monitor-output checks.
    void                 setPrimaries(ePrimaries source, ePrimaries target);
    void                 setupMonitor(eTransferFunction tf, int tonemapMode = 0, eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR);
    std::array<float, 4> readMonitor(const std::array<float, 4>& pixel, float windowAlpha = 1.0f);
    void                 checkHLGReference(bool encode);

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
    const EGLint CONFIG_ATTRS[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
    EGLConfig    config         = nullptr;
    EGLint       count          = 0;
    ASSERT_TRUE(eglChooseConfig(m_display, CONFIG_ATTRS, &config, 1, &count));
    if (count == 0)
        GTEST_SKIP() << "OpenGL ES 3 is unavailable";

    const EGLint CONTEXT_ATTRS[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    m_context                    = eglCreateContext(m_display, config, EGL_NO_CONTEXT, CONTEXT_ATTRS);
    ASSERT_NE(m_context, EGL_NO_CONTEXT);
    const EGLint SURFACE_ATTRS[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    m_surface                    = eglCreatePbufferSurface(m_display, config, SURFACE_ATTRS);
    ASSERT_NE(m_surface, EGL_NO_SURFACE);
    ASSERT_TRUE(eglMakeCurrent(m_display, m_surface, m_surface, m_context));

    const std::string EXTENSIONS = rc<const char*>(glGetString(GL_EXTENSIONS));
    if (!EXTENSIONS.contains("GL_EXT_color_buffer_float"))
        GTEST_SKIP() << "Floating-point framebuffer attachments are unavailable";

    glGenTextures(m_textures.size(), m_textures.data());
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    for (size_t i = 0; i < m_textures.size(); ++i) {
        glBindTexture(GL_TEXTURE_2D, m_textures.at(i));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, nullptr);
        if (i > 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i - 1, GL_TEXTURE_2D, m_textures.at(i), 0);
    }
    const GLenum ATTACHMENTS[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, ATTACHMENTS);
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

void CColorManagementMirrorTest::createProgram(bool mirror, bool tonemap, std::string_view fragment, eTransferFunction sourceTF, eTransferFunction targetTF, bool altTonemap) {
    glDeleteProgram(m_program);
    m_program                                                   = glCreateProgram();
    const std::array<std::pair<GLenum, std::string>, 2> SOURCES = {{
        {GL_VERTEX_SHADER,
         "#version 300 es\nout vec2 v_texcoord;\nvoid main() { v_texcoord = vec2(0.5); "
         "vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n"},
        {GL_FRAGMENT_SHADER, shaderSource(fragment, mirror, tonemap, sourceTF, targetTF, altTonemap)},
    }};
    for (const auto& [TYPE, SOURCE] : SOURCES) {
        const auto  SHADER = glCreateShader(TYPE);
        const auto* text   = SOURCE.c_str();
        glShaderSource(SHADER, 1, &text, nullptr);
        glCompileShader(SHADER);
        GLint                  compiled = GL_FALSE;
        std::array<char, 4096> log      = {};
        glGetShaderiv(SHADER, GL_COMPILE_STATUS, &compiled);
        glGetShaderInfoLog(SHADER, log.size(), nullptr, log.data());
        glAttachShader(m_program, SHADER);
        glDeleteShader(SHADER);
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

TEST_F(CColorManagementMirrorTest, WindowsPQBlackUsesHDRFloor) {
    ASSERT_NO_FATAL_FAILURE(setupMonitor(CM_TRANSFER_FUNCTION_ST2084_PQ));
    const auto& DESC = BT2100_IMAGE_DESCRIPTION->value();
    glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), DESC.getTFMinLuminance(), DESC.getTFMaxLuminance());
    for (float alpha : {0.0f, 0.25f, 1.0f}) {
        const auto RESULT = readMonitor({0.0f, 0.0f, 0.0f, alpha});
        for (size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(RESULT.at(i), HDR_MIN_LUMINANCE * alpha, 0.00001f);
    }
}

void CColorManagementMirrorTest::setupMonitor(eTransferFunction tf, int tonemapMode, eTransferFunction targetTF) {
    ASSERT_NO_FATAL_FAILURE(createProgram(false, tonemapMode != 0, "surface.frag", tf, targetTF, tonemapMode == 3));
    glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
    glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, tf == CM_TRANSFER_FUNCTION_HLG ? 1000.0f : tf == CM_TRANSFER_FUNCTION_EXT_LINEAR ? 80.0f : 10000.0f);
    glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), 0.0f, targetTF == CM_TRANSFER_FUNCTION_HLG ? 1000.0f : 500.0f);
    glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), 203.0f);
    glUniform1f(glGetUniformLocation(m_program, "dstRefLuminance"), 203.0f);
    glUniform1f(glGetUniformLocation(m_program, "maxLuminance"), 1000.0f);
    glUniform1f(glGetUniformLocation(m_program, "dstMaxLuminance"), 500.0f);
    glUniform1i(glGetUniformLocation(m_program, "tonemapMode"), tonemapMode);

    setPrimaries(CM_PRIMARIES_BT2020, CM_PRIMARIES_BT2020);
}

void CColorManagementMirrorTest::setPrimaries(ePrimaries source, ePrimaries target) {
    const auto SRC    = CPrimaries::from(source);
    const auto DST    = CPrimaries::from(target);
    auto       srcXYZ = SRC->toXYZ();
    auto       dstXYZ = DST->toXYZ();
    auto       matrix = SRC->convertMatrix(DST);
    const auto SRC_Y  = srcXYZ.mat().at(1);
    const auto DST_Y  = dstXYZ.mat().at(1);
    glUniform3f(glGetUniformLocation(m_program, "srcLumaCoeffs"), SRC_Y.at(0), SRC_Y.at(1), SRC_Y.at(2));
    glUniform3f(glGetUniformLocation(m_program, "dstLumaCoeffs"), DST_Y.at(0), DST_Y.at(1), DST_Y.at(2));
    const auto           CONVERT = matrix.mat();
    const auto           XYZ     = dstXYZ.mat();
    std::array<float, 9> convert = {}, xyz = {};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            convert.at(col * 3 + row) = CONVERT.at(row).at(col);
            xyz.at(col * 3 + row)     = XYZ.at(row).at(col);
        }
    }
    glUniformMatrix3fv(glGetUniformLocation(m_program, "convertMatrix"), 1, GL_FALSE, convert.data());
    glUniformMatrix3fv(glGetUniformLocation(m_program, "targetPrimariesXYZ"), 1, GL_FALSE, xyz.data());
    EXPECT_EQ(glGetError(), GL_NO_ERROR);
}

std::array<float, 4> CColorManagementMirrorTest::readMonitor(const std::array<float, 4>& pixel, float windowAlpha) {
    glBindTexture(GL_TEXTURE_2D, m_textures.at(0));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel.data());
    glUniform1f(glGetUniformLocation(m_program, "alpha"), windowAlpha);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    std::array<float, 4> result = {};
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, result.data());
    EXPECT_EQ(glGetError(), GL_NO_ERROR);
    return result;
}

void CColorManagementMirrorTest::checkHLGReference(bool encode) {
    // Independent BT.2100 reference values: inverse OETF, then
    // RGB_display = 1000 * RGB_scene * Y_scene^0.2, with BT.2020 Y weights
    // (0.2627, 0.6780, 0.0593). Zero black, 1000-nit peak, gamma 1.2.
    // EBU R 167 table 1.1 also gives 203 nits for the 75% neutral patch:
    // https://tech.ebu.ch/files/live/sites/tech/files/shared/r/r167.pdf
    struct SReference {
        std::array<float, 3> signal;
        std::array<float, 3> nits;
    };
    const std::array<SReference, 9> REFERENCES = {{
        {.signal = {0.0f, 0.0f, 0.0f}, .nits = {0.0f, 0.0f, 0.0f}},
        {.signal = {0.25f, 0.25f, 0.25f}, .nits = {9.605291f, 9.605291f, 9.605291f}},
        {.signal = {0.5f, 0.5f, 0.5f}, .nits = {50.697028f, 50.697028f, 50.697028f}},
        {.signal = {0.75f, 0.75f, 0.75f}, .nits = {203.152145f, 203.152145f, 203.152145f}},
        {.signal = {1.0f, 1.0f, 1.0f}, .nits = {1000.0f, 1000.0f, 1000.0f}},
        {.signal = {0.75f, 0.0f, 0.0f}, .nits = {155.493925f, 0.0f, 0.0f}},
        {.signal = {0.0f, 0.75f, 0.0f}, .nits = {0.0f, 187.960829f, 0.0f}},
        {.signal = {0.0f, 0.0f, 0.75f}, .nits = {0.0f, 0.0f, 115.460212f}},
        {.signal = {0.75f, 0.5f, 0.25f}, .nits = {175.460037f, 55.183909f, 13.795977f}},
    }};
    ASSERT_NO_FATAL_FAILURE(setupMonitor(encode ? CM_TRANSFER_FUNCTION_EXT_LINEAR : CM_TRANSFER_FUNCTION_HLG, 0, encode ? CM_TRANSFER_FUNCTION_HLG : CM_TRANSFER_FUNCTION_LINEAR));
    for (const auto& REFERENCE : REFERENCES) {
        SCOPED_TRACE(std::format("HLG signal {}, {}, {}", REFERENCE.signal.at(0), REFERENCE.signal.at(1), REFERENCE.signal.at(2)));
        for (float alpha : {0.0f, 0.25f, 0.5f, 1.0f}) {
            SCOPED_TRACE(std::format("alpha={}", alpha));
            const auto& INPUT    = encode ? REFERENCE.nits : REFERENCE.signal;
            const auto& EXPECTED = encode ? REFERENCE.signal : REFERENCE.nits;
            const float SCALE    = encode ? 80.0f : 1.0f;
            const auto  RESULT   = readMonitor({INPUT.at(0) * alpha / SCALE, INPUT.at(1) * alpha / SCALE, INPUT.at(2) * alpha / SCALE, alpha});
            for (size_t i = 0; i < 3; ++i)
                EXPECT_NEAR(RESULT.at(i), EXPECTED.at(i) * alpha, encode ? 0.0002f : 0.05f);
            EXPECT_NEAR(RESULT.at(3), alpha, 0.00001f);
        }
    }
}

TEST_F(CColorManagementMirrorTest, HLGMonitorMatchesReferenceDisplay) {
    checkHLGReference(false);
}

TEST_F(CColorManagementMirrorTest, HLGEncodingMatchesReferenceDisplay) {
    checkHLGReference(true);
}

TEST_F(CColorManagementMirrorTest, HLGUsesImagePrimaries) {
    struct SReference {
        ePrimaries           primaries;
        std::array<float, 3> luminance;
    };
    // Y rows of the D65 RGB-to-XYZ matrices (lin_sRGB_to_XYZ and lin_P3_to_XYZ):
    // https://www.w3.org/TR/css-color-4/#color-conversion-code
    // Keep these reference values independent of the production matrix calculation.
    const std::array<SReference, 2> REFERENCES = {{
        {CM_PRIMARIES_SRGB, {0.21263901f, 0.71516868f, 0.07219232f}},
        {CM_PRIMARIES_DISPLAY_P3, {0.22897456f, 0.69173852f, 0.07928691f}},
    }};
    // BT.2100-2 Table 5: inverse HLG OETF, (exp((signal - c) / a) + b) / 12.
    // b = 1 - 4a; c = 0.5 - a * ln(4a). The test signal is 75% encoded intensity.
    // https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-S!!PDF-E.pdf
    constexpr float HLG_A       = 0.17883277f;
    constexpr float HLG_B       = 0.28466892f;
    constexpr float HLG_C       = 0.55991073f;
    constexpr float TEST_SIGNAL = 0.75f;
    const float     SCENE       = (std::exp((TEST_SIGNAL - HLG_C) / HLG_A) + HLG_B) / 12.0f;
    for (const auto& REFERENCE : REFERENCES) {
        for (bool encode : {false, true}) {
            SCOPED_TRACE(std::format("primaries={}, encode={}", sc<int>(REFERENCE.primaries), encode));
            ASSERT_NO_FATAL_FAILURE(
                setupMonitor(encode ? CM_TRANSFER_FUNCTION_EXT_LINEAR : CM_TRANSFER_FUNCTION_HLG, 0, encode ? CM_TRANSFER_FUNCTION_HLG : CM_TRANSFER_FUNCTION_LINEAR));
            setPrimaries(REFERENCE.primaries, REFERENCE.primaries);
            for (size_t channel = 0; channel < 3; ++channel) {
                // Table 5 reference-display OOTF: 1000-nit peak, gamma 1.2 (exponent gamma - 1).
                const float NITS = 1000.0f * SCENE * std::pow(REFERENCE.luminance.at(channel) * SCENE, 0.2f);
                for (float alpha : {0.25f, 1.0f}) {
                    std::array<float, 4> pixel = {0, 0, 0, alpha};
                    pixel.at(channel)          = (encode ? NITS / 80.0f : TEST_SIGNAL) * alpha;
                    const auto RESULT          = readMonitor(pixel);
                    for (size_t i = 0; i < 3; ++i)
                        EXPECT_NEAR(RESULT.at(i), i == channel ? (encode ? TEST_SIGNAL : NITS) * alpha : 0.0f, encode ? 0.0002f : 0.05f);
                }
            }
        }
    }
}
