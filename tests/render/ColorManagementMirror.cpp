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

struct SShaderOptions {
    bool mirrorInput = false;
    bool motionBlur = false;
    bool colorManagement = true;
    bool blur            = false;
    bool blurMatte       = false;
};

static std::string shaderSource(std::string_view name, bool mirror, bool tonemap = false, eTransferFunction sourceTF = CM_TRANSFER_FUNCTION_SRGB,
                                eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR, bool altTonemap = false, bool sdrCapture = false, SShaderOptions options = {}) {
    if (name == "defines.h")
        return std::format("#define USE_CM {}\n#define USE_RGBA 1\n#define USE_MIRROR {}\n#define USE_TONEMAP {}\n"
                           "#define USE_BLUR {}\n#define USE_BLUR_MATTE {}\n#define USE_BLUR_ALPHA_MASK 0\n"
                           "#define USE_DISCARD 0\n#define USE_TINT 0\n#define USE_ROUNDING 0\n"
                           "#define USE_MOTION_BLUR {}\n#define USE_SDR_MOD 0\n#define USE_ICC 0\n#define USE_ALT_TONEMAP {}\n"
                           "#define SOURCE_TF {}\n#define TARGET_TF {}\n#define USE_SDR_CAPTURE {}\n#define USE_MIRROR_INPUT {}\n",
                           options.colorManagement ? 1 : 0, mirror ? 1 : 0, tonemap ? 1 : 0, options.blur ? 1 : 0, options.blurMatte ? 1 : 0, options.motionBlur ? 1 : 0,
                           altTonemap ? 1 : 0, sc<int>(sourceTF), sc<int>(targetTF), sdrCapture ? 1 : 0, options.mirrorInput ? 1 : 0);

    const auto IT = std::ranges::find(SHADERS, name, &std::pair<std::string_view, std::string_view>::first);
    if (IT == SHADERS.end())
        throw std::runtime_error(std::format("Missing shader {}", name));

    std::istringstream input{std::string{IT->second}};
    std::string        source, line;
    while (std::getline(input, line)) {
        if (line.starts_with("#include \""))
            source += shaderSource(std::string_view{line}.substr(10, line.size() - 11), mirror, tonemap, sourceTF, targetTF, altTonemap, sdrCapture, options);
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
                       eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR, bool altTonemap = false, bool sdrCapture = false, SShaderOptions options = {});
    void checkPixel(eTransferFunction tf, float reference, float sourceMax, float encoded, float expected, float alpha = 1.0f, bool tonemap = false, float capturePeak = 0.0f,
                    float redRatio = 1.0f, float sourceMin = 0.0f);

    // Linear monitor-output checks.
    void                 setPrimaries(ePrimaries source, ePrimaries target);
    void                 setupMonitor(eTransferFunction tf, int tonemapMode = 0, eTransferFunction targetTF = CM_TRANSFER_FUNCTION_LINEAR);
    std::array<float, 4> readMonitor(const std::array<float, 4>& pixel, float windowAlpha = 1.0f);
    void                 checkMonitorAlpha(int tonemapMode);
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

void CColorManagementMirrorTest::createProgram(bool mirror, bool tonemap, std::string_view fragment, eTransferFunction sourceTF, eTransferFunction targetTF, bool altTonemap,
                                               bool sdrCapture, SShaderOptions options) {
    glDeleteProgram(m_program);
    m_program                                                   = glCreateProgram();
    const std::array<std::pair<GLenum, std::string>, 2> SOURCES = {{
        {GL_VERTEX_SHADER,
         "#version 300 es\nout vec2 v_texcoord;\nvoid main() { v_texcoord = vec2(0.5); "
         "vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n"},
        {GL_FRAGMENT_SHADER, shaderSource(fragment, mirror, tonemap, sourceTF, targetTF, altTonemap, sdrCapture, options)},
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

void CColorManagementMirrorTest::checkPixel(eTransferFunction tf, float reference, float sourceMax, float encoded, float expected, float alpha, bool tonemap, float capturePeak,
                                            float redRatio, float sourceMin) {
    SCOPED_TRACE(std::format("tf={}, reference={}, encoded={}, alpha={}", sc<int>(tf), reference, encoded, alpha));
    std::array<float, 4> monitorWithMirror = {};
    for (bool mirror : {true, false}) {
        ASSERT_NO_FATAL_FAILURE(createProgram(mirror, tonemap, "surface.frag", tf));
        const std::array<float, 4> PIXEL = {encoded * alpha * redRatio, encoded * alpha, encoded * alpha, alpha};
        glBindTexture(GL_TEXTURE_2D, m_textures.at(0));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, PIXEL.data());
        glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
        glUniform1f(glGetUniformLocation(m_program, "alpha"), 1.0f);
        glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), reference);
        glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), capturePeak > 0.0f ? capturePeak : reference);
        glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), sourceMin, sourceMax);
        glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), 0.0f, 10000.0f);
        setPrimaries(CM_PRIMARIES_SRGB, CM_PRIMARIES_SRGB);
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
                EXPECT_NEAR(monitor.at(i), monitorWithMirror.at(i), 0.0001f);
            continue;
        }
        monitorWithMirror            = monitor;
        std::array<float, 4> capture = {};
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, capture.data());
        for (size_t i = 0; i < 3; ++i) {
            float expectedChannel = expected;
            if (i == 0 && redRatio != 1.0f) {
                // Color-ratio checks use linear sources: decode the expected SDR
                // value before applying the source's channel ratio.
                const float LINEAR = expected <= 0.04045f ? expected / 12.92f : std::pow((expected + 0.055f) / 1.055f, 2.4f);
                const float RED    = LINEAR * redRatio;
                expectedChannel    = RED <= 0.0031308f ? RED * 12.92f : (1.055f * std::pow(RED, 1.0f / 2.4f)) - 0.055f;
            }
            EXPECT_NEAR(capture.at(i), expectedChannel * alpha, 0.002f);
        }
        EXPECT_NEAR(capture.at(3), alpha, 0.0001f);
        EXPECT_EQ(glGetError(), GL_NO_ERROR);
    }
}

TEST_F(CColorManagementMirrorTest, PQReferenceWhiteAndMidtones) {
    for (float reference : {80.0f, 203.0f, 308.0f, 500.0f}) {
        for (float srgb : {0.0f, 0.125f, 0.25f, 0.5f, 0.75f, 1.0f}) {
            const float LINEAR = srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
            const float SCALED = std::pow(LINEAR * reference / 10000.0f, 0.1593017578125f);
            const float PQ     = std::pow((0.8359375f + (18.8515625f * SCALED)) / (1.0f + (18.6875f * SCALED)), 78.84375f);
            checkPixel(CM_TRANSFER_FUNCTION_ST2084_PQ, reference, 10000.0f, PQ, srgb);
        }
    }
}

TEST_F(CColorManagementMirrorTest, MatchingHDRFormatsPreserveMonitorAndConvertCapture) {
    const auto  DESCRIPTION = DEFAULT_HDR_IMAGE_DESCRIPTION;
    const auto& DESC        = DESCRIPTION->value();
    ASSERT_FALSE(DESCRIPTION->needsCM(DESCRIPTION));
    ASSERT_TRUE(DESCRIPTION->needsCM(DEFAULT_SRGB_IMAGE_DESCRIPTION));
    const std::array<std::array<float, 3>, 3> COLORS = {{
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.5f},
        {0.625f, 0.5f, 0.375f},
    }};
    for (const auto& COLOR : COLORS) {
        for (float alpha : {0.0f, 0.25f, 1.0f}) {
            for (float opacity : {0.4f, 1.0f}) {
                SCOPED_TRACE(std::format("red={}, alpha={}, opacity={}", COLOR.at(0), alpha, opacity));
                const std::array<float, 4>          PIXEL    = {COLOR.at(0) * alpha, COLOR.at(1) * alpha, COLOR.at(2) * alpha, alpha};
                std::array<std::array<float, 4>, 2> captures = {};
                for (size_t path = 0; path < captures.size(); ++path) {
                    // Both the matching PQ buffer and the linear sRGB work buffer
                    // must produce the same SDR capture.
                    const bool MATCHING = path == 0;
                    ASSERT_NO_FATAL_FAILURE(
                        createProgram(true, false, "surface.frag", CM_TRANSFER_FUNCTION_ST2084_PQ, MATCHING ? CM_TRANSFER_FUNCTION_ST2084_PQ : CM_TRANSFER_FUNCTION_LINEAR));
                    glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
                    glUniform1i(glGetUniformLocation(m_program, "captureOnly"), MATCHING);
                    glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), DESC.getTFMinLuminance(), DESC.getTFMaxLuminance());
                    glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), DESC.getTFMinLuminance(), DESC.getTFMaxLuminance());
                    glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), DESC.luminances.reference);
                    glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), DESC.getContentMaxLuminance());
                    setPrimaries(CM_PRIMARIES_BT2020, MATCHING ? CM_PRIMARIES_BT2020 : CM_PRIMARIES_SRGB);
                    const auto MONITOR = readMonitor(PIXEL, opacity);
                    if (MATCHING) {
                        for (size_t channel = 0; channel < 4; ++channel)
                            EXPECT_FLOAT_EQ(MONITOR.at(channel), PIXEL.at(channel) * opacity);
                    }
                    glReadBuffer(GL_COLOR_ATTACHMENT1);
                    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, captures.at(path).data());
                }
                for (size_t channel = 0; channel < 4; ++channel)
                    EXPECT_NEAR(captures.at(0).at(channel), captures.at(1).at(channel), 0.002f);
                EXPECT_NEAR(captures.at(0).at(3), alpha * opacity, 0.0001f);
                EXPECT_EQ(glGetError(), GL_NO_ERROR);
                if (COLOR.at(0) != 0.5f)
                    continue;
                // PQ 0.5 is about 92.25 nits, or 0.7045 sRGB at 203-nit white.
                EXPECT_NEAR(captures.at(0).at(0), 0.7045f * alpha * opacity, 0.002f);
            }
        }
    }
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

TEST_F(CColorManagementMirrorTest, LinearHDRReferenceWhiteAndTransparency) {
    for (float alpha : {0.0f, 0.25f, 0.5f, 1.0f})
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 80.0f, 308.0f / 80.0f, 1.0f, alpha);
}

TEST_F(CColorManagementMirrorTest, HLGReferenceWhite) {
    // Invert the reference-display OOTF before applying the HLG OETF.
    const float SCENE = std::pow(0.308f, 1.0f / 1.2f);
    const float HLG   = (0.17883277f * std::log((12.0f * SCENE) - 0.28466892f)) + 0.55991073f;
    checkPixel(CM_TRANSFER_FUNCTION_HLG, 308.0f, 1000.0f, HLG, 1.0f);
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
    auto                 captureMatrix = DST->convertMatrix(CPrimaries::from(CM_PRIMARIES_SRGB));
    const auto           CAPTURE       = captureMatrix.mat();
    std::array<float, 9> convert = {}, xyz = {}, capture = {};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            convert.at(col * 3 + row) = CONVERT.at(row).at(col);
            xyz.at(col * 3 + row)     = XYZ.at(row).at(col);
            capture.at(col * 3 + row) = CAPTURE.at(row).at(col);
        }
    }
    glUniformMatrix3fv(glGetUniformLocation(m_program, "convertMatrix"), 1, GL_FALSE, convert.data());
    glUniformMatrix3fv(glGetUniformLocation(m_program, "blurConvertMatrix"), 1, GL_FALSE, convert.data());
    glUniform3f(glGetUniformLocation(m_program, "blurLumaCoeffs"), SRC_Y.at(0), SRC_Y.at(1), SRC_Y.at(2));
    glUniformMatrix3fv(glGetUniformLocation(m_program, "captureMatrix"), 1, GL_FALSE, capture.data());
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

void CColorManagementMirrorTest::checkMonitorAlpha(int tonemapMode) {
    for (auto tf : {CM_TRANSFER_FUNCTION_ST2084_PQ, CM_TRANSFER_FUNCTION_EXT_LINEAR}) {
        ASSERT_NO_FATAL_FAILURE(setupMonitor(tf, tonemapMode));
        for (const auto& NITS : {std::array{0.0f, 0.0f, 0.0f}, std::array{50.0f, 50.0f, 50.0f}, std::array{1000.0f, 1000.0f, 1000.0f}, std::array{1000.0f, 400.0f, 100.0f}}) {
            SCOPED_TRACE(std::format("mode={}, tf={}, RGB nits={}, {}, {}", tonemapMode, sc<int>(tf), NITS.at(0), NITS.at(1), NITS.at(2)));
            std::array<float, 4> pixel = {0.0f, 0.0f, 0.0f, 1.0f};
            for (size_t i = 0; i < 3; ++i) {
                if (tf == CM_TRANSFER_FUNCTION_EXT_LINEAR) {
                    pixel.at(i) = NITS.at(i) / 80.0f;
                    continue;
                }
                const double P = std::pow(NITS.at(i) / 10000.0, 2610.0 / 16384.0);
                pixel.at(i)    = std::pow(((3424.0 / 4096.0) + (2413.0 / 128.0 * P)) / (1.0 + (2392.0 / 128.0 * P)), 2523.0 / 32.0);
            }
            const auto OPAQUE = readMonitor(pixel);
            for (size_t i = 0; i < 3; ++i) {
                ASSERT_TRUE(std::isfinite(OPAQUE.at(i)));
                ASSERT_TRUE(NITS.at(i) == 0.0f || OPAQUE.at(i) > 0.0f);
                if (tonemapMode == 0)
                    continue;
                EXPECT_LE(OPAQUE.at(i), 500.05f);
            }
            for (float sourceAlpha : {0.0f, 0.25f, 0.5f, 1.0f}) {
                for (float windowAlpha : {0.0f, 0.25f, 0.5f, 1.0f}) {
                    SCOPED_TRACE(std::format("source alpha={}, window alpha={}", sourceAlpha, windowAlpha));
                    const auto RESULT = readMonitor({pixel.at(0) * sourceAlpha, pixel.at(1) * sourceAlpha, pixel.at(2) * sourceAlpha, sourceAlpha}, windowAlpha);
                    // Compare against an opaque invocation of the real shader;
                    // do not duplicate its tone curve as the expected result.
                    for (size_t i = 0; i < 3; ++i)
                        EXPECT_NEAR(RESULT.at(i), OPAQUE.at(i) * sourceAlpha * windowAlpha, 0.1f);
                    EXPECT_NEAR(RESULT.at(3), sourceAlpha * windowAlpha, 0.00001f);
                }
            }
        }
    }
}

TEST_F(CColorManagementMirrorTest, LinearMonitorPreservesAlpha) {
    checkMonitorAlpha(0);
}

TEST_F(CColorManagementMirrorTest, TonemappedMonitorPreservesAlpha) {
    checkMonitorAlpha(1);
}

TEST_F(CColorManagementMirrorTest, ClippedMonitorPreservesAlpha) {
    checkMonitorAlpha(2);
}

TEST_F(CColorManagementMirrorTest, AlternateTonemappedMonitorPreservesAlpha) {
    checkMonitorAlpha(3);
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
        for (auto targetTF : {CM_TRANSFER_FUNCTION_LINEAR, CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_ST2084_PQ, CM_TRANSFER_FUNCTION_HLG}) {
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
    const float ENCODED = (1.055f * std::pow(308.0f / 80.0f, 1.0f / 2.4f)) - 0.055f;
    checkPixel(CM_TRANSFER_FUNCTION_EXT_SRGB, 308.0f, 80.0f, ENCODED, 1.0f);
}

TEST_F(CColorManagementMirrorTest, MonitorTonemappingDoesNotChangeCapture) {
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 80.0f, 308.0f / 80.0f, 1.0f, 1.0f, true);
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 80.0f, 308.0f * 0.21404114f / 80.0f, 0.5f, 1.0f, true);
}

TEST_F(CColorManagementMirrorTest, HDRHighlightsRemainDistinctAndPreserveHue) {
    // At 4x reference white, the peak is white. Intermediate highlights retain
    // detail, while a 50% sRGB midtone is unchanged by the shoulder.
    for (float alpha : {0.0f, 0.25f, 0.5f, 1.0f}) {
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.0f, 80.0f, 250.0f * 0.21404114f / 80.0f, 0.5f, alpha, false, 1000.0f);
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.0f, 80.0f, 250.0f / 80.0f, 0.9452769f, alpha, false, 1000.0f, 0.5f);
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.0f, 80.0f, 500.0f / 80.0f, 0.98785897f, alpha, false, 1000.0f, 0.5f);
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.0f, 80.0f, 1000.0f / 80.0f, 1.0f, alpha, false, 1000.0f, 0.5f);
    }
}

TEST_F(CColorManagementMirrorTest, SDRCaptureDoesNotUseHDRShoulder) {
    checkPixel(CM_TRANSFER_FUNCTION_SRGB, 80.0f, 80.0f, 1.0f, 1.0f, 1.0f, false, 1000.0f);
}

TEST_F(CColorManagementMirrorTest, ParametricLinearWhiteUsesDeclaredEncodingRange) {
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 308.0f, 1.0f, 1.0f);
    checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 308.0f, 1000.0f, 0.308f, 1.0f);
}

TEST_F(CColorManagementMirrorTest, ExtendedSDRCaptureUsesProductionPeak) {
    for (auto tf : {CM_TRANSFER_FUNCTION_EXT_LINEAR, CM_TRANSFER_FUNCTION_EXT_SRGB}) {
        for (uint32_t reference : {80U, 308U}) {
            const SImageDescription DESC{
                .transferFunction = tf,
                .luminances       = {.min = 0, .max = reference, .reference = reference},
            };
            for (float alpha : {0.25f, 1.0f})
                checkPixel(tf, reference, DESC.getTFMaxLuminance(), 1.0f, 1.0f, alpha, false, DESC.getContentMaxLuminance());
        }
    }
}

TEST_F(CColorManagementMirrorTest, ExtendedSDRCapturePreservesBlackWhiteAndMidtones) {
    for (auto tf : {CM_TRANSFER_FUNCTION_EXT_SRGB, CM_TRANSFER_FUNCTION_EXT_LINEAR}) {
        for (uint32_t reference : {80U, 308U}) {
            const SImageDescription DESC{
                .transferFunction = tf,
                .luminances       = {.min = 0.2f, .max = reference, .reference = reference},
            };
            for (float srgb : {0.0f, 0.5f, 1.0f}) {
                const float LINEAR  = srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
                const float ENCODED = tf == CM_TRANSFER_FUNCTION_EXT_LINEAR ? LINEAR : srgb;
                for (float alpha : {0.0f, 0.25f, 1.0f})
                    checkPixel(tf, reference, DESC.getTFMaxLuminance(), ENCODED, srgb, alpha, false, DESC.getContentMaxLuminance(), 1.0f, DESC.getTFMinLuminance());
            }
        }
    }
}

TEST_F(CColorManagementMirrorTest, CaptureShoulderUsesLuminanceAboveBlack) {
    // Match the zero-black highlight test with all luminances offset by 0.2 nits.
    for (float alpha : {0.0f, 0.25f, 1.0f}) {
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.2f, 80.2f, 0.0f, 0.0f, alpha, false, 1000.2f, 1.0f, 0.2f);
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.2f, 80.2f, 250.0f / 80.0f, 0.9452769f, alpha, false, 1000.2f, 1.0f, 0.2f);
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.2f, 80.2f, 500.0f / 80.0f, 0.98785897f, alpha, false, 1000.2f, 1.0f, 0.2f);
        checkPixel(CM_TRANSFER_FUNCTION_EXT_LINEAR, 250.2f, 80.2f, 1000.0f / 80.0f, 1.0f, alpha, false, 1000.2f, 1.0f, 0.2f);
    }
}

TEST_F(CColorManagementMirrorTest, GammaCaptureUsesReferenceWhiteWithEncodingHeadroom) {
    // Four times reference white fits in the encoding; the capture shoulder
    // matches HDRHighlightsRemainDistinctAndPreserveHue for every transfer function.
    const std::array<std::pair<float, float>, 5> PATCHES = {{
        {0.0f, 0.0f},
        {0.21404114f, 0.5f},
        {1.0f, 0.9452769f},
        {2.0f, 0.98785897f},
        {4.0f, 1.0f},
    }};
    for (auto tf : {CM_TRANSFER_FUNCTION_GAMMA22, CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_EXT_LINEAR}) {
        for (float black : {0.0f, 0.2f}) {
            for (float scale : {1.0f, 2.5f}) {
                for (const auto& [LINEAR, EXPECTED] : PATCHES) {
                    float encoded = LINEAR / 4.0f;
                    if (tf == CM_TRANSFER_FUNCTION_GAMMA22)
                        encoded = std::pow(encoded, 1.0f / 2.2f);
                    else if (tf == CM_TRANSFER_FUNCTION_SRGB)
                        encoded = encoded <= 0.0031308f ? 12.92f * encoded : 1.055f * std::pow(encoded, 1.0f / 2.4f) - 0.055f;
                    for (float alpha : {0.0f, 0.25f, 1.0f})
                        checkPixel(tf, black + 250.0f * scale, black + 1000.0f * scale, encoded, EXPECTED, alpha, false, black + 1000.0f * scale, 1.0f, black);
                }
            }
        }
    }
}

TEST_F(CColorManagementMirrorTest, WindowCaptureMatchesMonitorCapture) {
    struct SSource {
        eTransferFunction tf        = CM_TRANSFER_FUNCTION_SRGB;
        float             min       = 0.0f;
        float             max       = 80.0f;
        float             reference = 80.0f;
        float             peak      = 80.0f;
    };
    const std::array<SSource, 7> SOURCES = {{
        {CM_TRANSFER_FUNCTION_EXT_LINEAR, 0.0f, 80.0f, 308.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_EXT_SRGB, 0.2f, 80.0f, 308.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_ST2084_PQ, 0.0f, 10000.0f, 203.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_HLG, 0.0f, 1000.0f, 203.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_GAMMA22, 0.2f, 600.0f, 203.0f, 600.0f},
        {CM_TRANSFER_FUNCTION_SRGB, 0.0f, 600.0f, 203.0f, 600.0f},
        {CM_TRANSFER_FUNCTION_SRGB, 0.2f, 80.0f, 80.0f, 80.0f},
    }};
    for (const auto& SOURCE : SOURCES) {
        SCOPED_TRACE(sc<int>(SOURCE.tf));
        for (float encoded : {0.0f, 0.25f, 0.5f, 1.0f}) {
            for (float alpha : {0.0f, 0.25f, 1.0f}) {
                for (float opacity : {0.4f, 1.0f}) {
                    std::array<std::array<float, 4>, 2> captures = {};
                    for (size_t path = 0; path < captures.size(); ++path) {
                        const bool WINDOW = path == 1;
                        ASSERT_NO_FATAL_FAILURE(
                            createProgram(!WINDOW, false, "surface.frag", SOURCE.tf, WINDOW ? CM_TRANSFER_FUNCTION_SRGB : CM_TRANSFER_FUNCTION_LINEAR, false, WINDOW));
                        glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
                        glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), SOURCE.min, SOURCE.max);
                        glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), WINDOW ? 0.2f : 0.0f, WINDOW ? 80.0f : 10000.0f);
                        glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), SOURCE.reference);
                        glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), SOURCE.peak);
                        setPrimaries(CM_PRIMARIES_SRGB, CM_PRIMARIES_SRGB);
                        captures.at(path) = readMonitor({encoded * alpha, encoded * alpha * 0.75f, encoded * alpha * 0.5f, alpha}, opacity);
                        if (!WINDOW) {
                            glReadBuffer(GL_COLOR_ATTACHMENT1);
                            glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, captures.at(path).data());
                        }
                    }
                    for (size_t channel = 0; channel < 4; ++channel)
                        EXPECT_NEAR(captures.at(0).at(channel), captures.at(1).at(channel), 0.002f);
                    EXPECT_NEAR(captures.at(1).at(3), alpha * opacity, 0.0001f);
                    EXPECT_EQ(glGetError(), GL_NO_ERROR);
                }
            }
        }
    }
}

TEST_F(CColorManagementMirrorTest, IntermediateBuffersPreserveCaptureWhiteAndOpacity) {
    GLuint captureTexture = 0;
    glGenTextures(1, &captureTexture);
    for (float sourceAlpha : {0.0f, 0.25f, 1.0f}) {
        ASSERT_NO_FATAL_FAILURE(createProgram(true, false, "surface.frag", CM_TRANSFER_FUNCTION_GAMMA22, CM_TRANSFER_FUNCTION_EXT_LINEAR));
        glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
        glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, 308.0f);
        glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), 0.0f, 80.0f);
        glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), 308.0f);
        glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), 308.0f);
        setPrimaries(CM_PRIMARIES_SRGB, CM_PRIMARIES_SRGB);
        const float ENCODED = std::pow(0.25f, 1.0f / 2.2f) * sourceAlpha;
        const auto MONITOR = readMonitor({ENCODED, ENCODED, ENCODED, sourceAlpha});
        std::array<float, 4> capture = {};
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, capture.data());
        EXPECT_NEAR(capture.at(0), 0.5370987f * sourceAlpha, 0.002f);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, captureTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, capture.data());
        glActiveTexture(GL_TEXTURE0);

        // Intermediate buffers carry the monitor's reference (203), not the
        // client's (308). Their already-converted capture must pass through.
        for (bool cm : {false, true}) {
            for (bool motionBlur : {false, true}) {
                ASSERT_NO_FATAL_FAILURE(createProgram(true, false, "surface.frag", CM_TRANSFER_FUNCTION_EXT_LINEAR, CM_TRANSFER_FUNCTION_EXT_LINEAR, false, false,
                                                      {.mirrorInput = true, .motionBlur = motionBlur, .colorManagement = cm}));
                glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
                glUniform1i(glGetUniformLocation(m_program, "mirrorTex"), 1);
                glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, 80.0f);
                glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), 0.0f, 80.0f);
                glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), 203.0f);
                glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), 1000.0f);
                setPrimaries(CM_PRIMARIES_SRGB, CM_PRIMARIES_SRGB);
                for (auto name : {"motionPrevBox", "motionCurrBox", "motionSourceBox"})
                    glUniform4f(glGetUniformLocation(m_program, name), 0.0f, 0.0f, 1.0f, 1.0f);
                glUniform2f(glGetUniformLocation(m_program, "motionSourceTexOrigin"), 0.0f, 0.0f);
                glUniform2f(glGetUniformLocation(m_program, "motionSourceTexSize"), 1.0f, 1.0f);
                glUniform1i(glGetUniformLocation(m_program, "motionSamples"), 2);
                for (float opacity : {0.25f, 1.0f}) {
                    const auto RESULT = readMonitor(MONITOR, opacity);
                    std::array<float, 4> captured = {};
                    glReadBuffer(GL_COLOR_ATTACHMENT1);
                    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, captured.data());
                    for (size_t channel = 0; channel < 4; ++channel) {
                        EXPECT_NEAR(RESULT.at(channel), MONITOR.at(channel) * opacity, 0.002f);
                        EXPECT_NEAR(captured.at(channel), capture.at(channel) * opacity, 0.002f);
                    }
                }
            }
        }
    }
    glDeleteTextures(1, &captureTexture);
    EXPECT_EQ(glGetError(), GL_NO_ERROR);
}

TEST_F(CColorManagementMirrorTest, WindowCaptureSurvivesSDRIntermediateBuffer) {
    for (float sourceAlpha : {0.0f, 0.25f, 1.0f}) {
        ASSERT_NO_FATAL_FAILURE(createProgram(false, false, "surface.frag", CM_TRANSFER_FUNCTION_GAMMA22, CM_TRANSFER_FUNCTION_SRGB, false, true));
        glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
        glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, 308.0f);
        glUniform2f(glGetUniformLocation(m_program, "dstTFRange"), 0.2f, 80.0f);
        glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), 308.0f);
        glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), 308.0f);
        setPrimaries(CM_PRIMARIES_SRGB, CM_PRIMARIES_SRGB);
        const float ENCODED = std::pow(0.25f, 1.0f / 2.2f) * sourceAlpha;
        const auto  CAPTURE = readMonitor({ENCODED, ENCODED, ENCODED, sourceAlpha});
        EXPECT_NEAR(CAPTURE.at(0), 0.5370987f * sourceAlpha, 0.002f);

        // Matching SDR buffer descriptions skip CM on the final composite.
        ASSERT_NO_FATAL_FAILURE(createProgram(false, false, "surface.frag", CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_SRGB, false, false, {.colorManagement = false}));
        glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
        for (float opacity : {0.25f, 1.0f}) {
            const auto RESULT = readMonitor(CAPTURE, opacity);
            for (size_t channel = 0; channel < 4; ++channel)
                EXPECT_NEAR(RESULT.at(channel), CAPTURE.at(channel) * opacity, 0.002f);
        }
    }
}

TEST_F(CColorManagementMirrorTest, BlurCaptureMatchesSourceConversionWithoutChangingMonitor) {
    struct SSource {
        eTransferFunction tf        = CM_TRANSFER_FUNCTION_EXT_LINEAR;
        float             min       = 0.0f;
        float             max       = 80.0f;
        float             reference = 203.0f;
        float             peak      = 1000.0f;
    };
    const std::array<SSource, 6> SOURCES    = {{
        {CM_TRANSFER_FUNCTION_EXT_LINEAR, 0.0f, 80.0f, 80.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_EXT_LINEAR, 0.0f, 80.0f, 500.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_ST2084_PQ, 0.0f, 10000.0f, 203.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_HLG, 0.0f, 1000.0f, 203.0f, 1000.0f},
        {CM_TRANSFER_FUNCTION_GAMMA22, 0.2f, 80.0f, 80.0f, 80.0f},
        {CM_TRANSFER_FUNCTION_SRGB, 0.2f, 80.0f, 80.0f, 80.0f},
    }};
    GLuint                       background = 0;
    glGenTextures(1, &background);
    for (const auto& SOURCE : SOURCES) {
        for (auto primaries : {CM_PRIMARIES_SRGB, CM_PRIMARIES_BT2020}) {
            const float                ENCODED = SOURCE.tf == CM_TRANSFER_FUNCTION_EXT_LINEAR ? 0.21404114f * SOURCE.reference / 80.0f : 0.5f;
            const std::array<float, 4> PIXEL   = {ENCODED, ENCODED * 0.75f, ENCODED * 0.5f, 1.0f};
            ASSERT_NO_FATAL_FAILURE(createProgram(true, false, "surface.frag", SOURCE.tf));
            setPrimaries(primaries, CM_PRIMARIES_SRGB);
            glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), SOURCE.min, SOURCE.max);
            glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), SOURCE.reference);
            glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), SOURCE.peak);
            readMonitor(PIXEL);
            std::array<float, 4> reference = {};
            glReadBuffer(GL_COLOR_ATTACHMENT1);
            glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, reference.data());
            if (primaries == CM_PRIMARIES_SRGB && SOURCE.tf == CM_TRANSFER_FUNCTION_EXT_LINEAR) {
                EXPECT_NEAR(reference.at(0), 0.5f, 0.002f);
            }

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, background);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, PIXEL.data());
            glActiveTexture(GL_TEXTURE0);

            for (bool cm : {false, true}) {
                for (bool matte : {false, true}) {
                    SCOPED_TRACE(std::format("tf={}, primaries={}, cm={}, matte={}", sc<int>(SOURCE.tf), sc<int>(primaries), cm, matte));
                    ASSERT_NO_FATAL_FAILURE(createProgram(true, false, "surface.frag", CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_LINEAR, false, false,
                                                          {.mirrorInput = matte, .colorManagement = cm, .blur = true, .blurMatte = matte}));
                    setPrimaries(primaries, CM_PRIMARIES_SRGB);
                    glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, 80.0f);
                    glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), 80.0f);
                    glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), 80.0f);
                    glUniform1i(glGetUniformLocation(m_program, "blurredBG"), 1);
                    glUniform2f(glGetUniformLocation(m_program, "uvSize"), 1.0f, 1.0f);
                    glUniform1i(glGetUniformLocation(m_program, "blurTF"), SOURCE.tf);
                    glUniform2f(glGetUniformLocation(m_program, "blurTFRange"), SOURCE.min, SOURCE.max);
                    glUniform1f(glGetUniformLocation(m_program, "blurReferenceLuminance"), SOURCE.reference);
                    glUniform1f(glGetUniformLocation(m_program, "blurMaxLuminance"), SOURCE.peak);
                    glUniform1i(glGetUniformLocation(m_program, "blurAlphaMatte"), 1);
                    glUniform1f(glGetUniformLocation(m_program, "blurAlpha"), 0.5f);
                    // Reuse the background's red channel as a fractional blur matte.
                    const float MASK = matte ? std::min(ENCODED, 1.0f) * 0.5f : 1.0f;
                    for (float alpha : {0.0f, 0.25f, 1.0f}) {
                        for (float opacity : {0.4f, 1.0f}) {
                            glUniform1i(glGetUniformLocation(m_program, "blurCaptureCM"), 0);
                            const auto BEFORE = readMonitor({0.0f, 0.0f, 0.0f, alpha}, opacity);
                            glUniform1i(glGetUniformLocation(m_program, "blurCaptureCM"), 1);
                            const auto           AFTER   = readMonitor({0.0f, 0.0f, 0.0f, alpha}, opacity);
                            std::array<float, 4> capture = {};
                            glReadBuffer(GL_COLOR_ATTACHMENT1);
                            glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, capture.data());
                            for (size_t channel = 0; channel < 4; ++channel)
                                EXPECT_FLOAT_EQ(BEFORE.at(channel), AFTER.at(channel));
                            for (size_t channel = 0; channel < 3; ++channel)
                                EXPECT_NEAR(capture.at(channel), reference.at(channel) * (1.0f - alpha * opacity) * MASK, 0.002f);
                            EXPECT_NEAR(capture.at(3), alpha * opacity + (1.0f - alpha * opacity) * MASK, 0.001f);
                            EXPECT_EQ(glGetError(), GL_NO_ERROR);
                        }
                    }
                }
            }
        }
    }
    glDeleteTextures(1, &background);
}

TEST_F(CColorManagementMirrorTest, WindowCaptureConvertsBlurBackground) {
    ASSERT_NO_FATAL_FAILURE(createProgram(false, false, "surface.frag", CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_SRGB, false, true, {.blur = true}));
    setPrimaries(CM_PRIMARIES_SRGB, CM_PRIMARIES_SRGB);
    glUniform2f(glGetUniformLocation(m_program, "srcTFRange"), 0.0f, 80.0f);
    glUniform1f(glGetUniformLocation(m_program, "srcRefLuminance"), 80.0f);
    glUniform1f(glGetUniformLocation(m_program, "captureMaxLuminance"), 80.0f);
    glUniform1i(glGetUniformLocation(m_program, "blurredBG"), 1);
    glUniform2f(glGetUniformLocation(m_program, "uvSize"), 1.0f, 1.0f);
    glUniform1i(glGetUniformLocation(m_program, "blurCaptureCM"), 1);
    glUniform1i(glGetUniformLocation(m_program, "blurTF"), CM_TRANSFER_FUNCTION_EXT_LINEAR);
    glUniform2f(glGetUniformLocation(m_program, "blurTFRange"), 0.0f, 80.0f);
    glUniform1f(glGetUniformLocation(m_program, "blurReferenceLuminance"), 203.0f);
    glUniform1f(glGetUniformLocation(m_program, "blurMaxLuminance"), 1000.0f);

    GLuint background = 0;
    glGenTextures(1, &background);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, background);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    const float                ENCODED = 0.21404114f * 203.0f / 80.0f;
    const std::array<float, 4> PIXEL   = {ENCODED, ENCODED, ENCODED, 1.0f};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, PIXEL.data());
    glActiveTexture(GL_TEXTURE0);
    for (float alpha : {0.0f, 0.25f, 1.0f}) {
        for (float opacity : {0.4f, 1.0f}) {
            const auto CAPTURE = readMonitor({0.0f, 0.0f, 0.0f, alpha}, opacity);
            for (size_t channel = 0; channel < 3; ++channel)
                EXPECT_NEAR(CAPTURE.at(channel), 0.5f * (1.0f - alpha * opacity), 0.002f);
            EXPECT_NEAR(CAPTURE.at(3), 1.0f, 0.0001f);
        }
    }
    glDeleteTextures(1, &background);
}
