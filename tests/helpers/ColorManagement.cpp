#include <helpers/cm/ColorManagement.hpp>
#include <render/types.hpp>

#include <gtest/gtest.h>

using namespace NColorManagement;

static RGBAColor convertWithoutTonemapping(RGBAColor color, PImageDescription source, PImageDescription target) {
    const Render::SCMSettings settings = {
        .sourceTF   = source->value().transferFunction,
        .targetTF   = target->value().transferFunction,
        .srcTFRange = {.min = source->value().getTFMinLuminance(), .max = source->value().getTFMaxLuminance()},
        .dstTFRange = {.min = target->value().getTFMinLuminance(), .max = target->value().getTFMaxLuminance()},
    };
    return convertColor(color, source, target, settings);
}

static void expectNeutralColor(const RGBAColor& color, double luminance, double alpha = 1.0) {
    EXPECT_NEAR(color.c.r, luminance, 0.0001);
    EXPECT_NEAR(color.c.g, luminance, 0.0001);
    EXPECT_NEAR(color.c.b, luminance, 0.0001);
    EXPECT_DOUBLE_EQ(color.c.a, alpha);
}

static PImageDescription nativeLinearDescription(const SImageDescription::SPCLuminances& luminances) {
    return CImageDescription::from(SImageDescription{
        .transferFunction = CM_TRANSFER_FUNCTION_EXT_LINEAR,
        .primariesNameSet = true,
        .primariesNamed   = CM_PRIMARIES_SRGB,
        .luminances       = luminances,
    });
}

TEST(ColorManagement, WindowsScRGBDecodesReferenceWhite) {
    const auto color = convertWithoutTonemapping({{.r = 2.5375, .g = 2.5375, .b = 2.5375, .a = 1.0}}, SCRGB_IMAGE_DESCRIPTION, LINEAR_NN_IMAGE_DESCRIPTION);
    expectNeutralColor(color, 203.0);
}

TEST(ColorManagement, WorkBufferPreservesScRGBEncodingWithMonitorMetadata) {
    for (const auto& luminances : {
             SImageDescription::SPCLuminances{.min = 0.2f, .max = 80, .reference = 80},
             SImageDescription::SPCLuminances{.min = 0.005f, .max = 600, .reference = 203},
             SImageDescription::SPCLuminances{.min = 0.0f, .max = 10000, .reference = 203},
         }) {
        const auto workBuffer = LINEAR_IMAGE_DESCRIPTION->with(luminances);
        EXPECT_EQ(workBuffer->value().luminances, luminances);

        const auto encoded = convertWithoutTonemapping({{.r = 203.0, .g = 203.0, .b = 203.0, .a = 1.0}}, LINEAR_NN_IMAGE_DESCRIPTION, workBuffer);
        expectNeutralColor(encoded, 2.5375);

        const auto decoded = convertWithoutTonemapping(encoded, workBuffer, LINEAR_NN_IMAGE_DESCRIPTION);
        expectNeutralColor(decoded, 203.0);
    }
}

TEST(ColorManagement, EquivalentEncodingsDoNotRequireConversion) {
    const auto workBuffer = LINEAR_IMAGE_DESCRIPTION->with({.min = 0.0f, .max = 80, .reference = 203});
    auto       native     = workBuffer->value();
    native.isWindows      = false;
    const auto source     = CImageDescription::from(native);

    ASSERT_NE(source->id(), workBuffer->id());
    EXPECT_FALSE(source->needsCM(workBuffer));
    EXPECT_FALSE(workBuffer->needsCM(source));
    EXPECT_FALSE(workBuffer->needsCM(workBuffer));
}

TEST(ColorManagement, NativeLinearReferenceWhiteUsesDeclaredRange) {
    const auto source     = nativeLinearDescription({.min = 0.0f, .max = 10000, .reference = 203});
    const auto workBuffer = LINEAR_IMAGE_DESCRIPTION->with(source->value().luminances);

    const auto encoded = convertWithoutTonemapping({{.r = 0.0203, .g = 0.0203, .b = 0.0203, .a = 1.0}}, source, workBuffer);
    expectNeutralColor(encoded, 2.5375);

    const auto decoded = convertWithoutTonemapping(encoded, workBuffer, LINEAR_NN_IMAGE_DESCRIPTION);
    expectNeutralColor(decoded, 203.0);
}

TEST(ColorManagement, MatchingMetadataWithDifferentEncodingRequiresConversion) {
    for (const auto& luminances : {
             SImageDescription::SPCLuminances{.min = 0.2f, .max = 80, .reference = 80},
             SImageDescription::SPCLuminances{.min = 0.0f, .max = 10000, .reference = 203},
         }) {
        const auto source     = nativeLinearDescription(luminances);
        const auto workBuffer = LINEAR_IMAGE_DESCRIPTION->with(luminances);

        ASSERT_EQ(source->value().luminances, workBuffer->value().luminances);
        ASSERT_EQ(source->value().transferFunction, workBuffer->value().transferFunction);
        ASSERT_EQ(source->value().getPrimaries(), workBuffer->value().getPrimaries());
        EXPECT_TRUE(source->needsCM(workBuffer));
        EXPECT_TRUE(workBuffer->needsCM(source));
    }
}

TEST(ColorManagement, NativeLinearDecodingIncludesMinimumLuminance) {
    const auto source = nativeLinearDescription({.min = 1.0f, .max = 1001, .reference = 101});
    for (const auto value : {0.0, 0.1, 1.0}) {
        const auto decoded = convertWithoutTonemapping({{.r = value, .g = value, .b = value, .a = 1.0}}, source, LINEAR_NN_IMAGE_DESCRIPTION);
        expectNeutralColor(decoded, 1.0 + value * 1000.0);
    }
}

TEST(ColorManagement, NativeLinearConversionPreservesPremultipliedAlpha) {
    const auto source = nativeLinearDescription({.min = 1.0f, .max = 1001, .reference = 101});
    for (const auto alpha : {0.0, 0.25, 0.5, 1.0}) {
        const double value   = 0.1 * alpha;
        const auto   encoded = convertWithoutTonemapping({{.r = value, .g = value, .b = value, .a = alpha}}, source, LINEAR_IMAGE_DESCRIPTION);
        expectNeutralColor(encoded, 101.0 / 80.0 * alpha, alpha);

        const auto roundTrip = convertWithoutTonemapping(encoded, LINEAR_IMAGE_DESCRIPTION, source);
        expectNeutralColor(roundTrip, value, alpha);
    }
}

TEST(ColorManagement, NativeLinearConversionPreservesExtendedRange) {
    const auto source = nativeLinearDescription({.min = 0.0f, .max = 10000, .reference = 203});
    const auto color  = convertWithoutTonemapping({{.r = -0.0001, .g = 0.0203, .b = 1.25, .a = 1.0}}, source, LINEAR_NN_IMAGE_DESCRIPTION);
    EXPECT_NEAR(color.c.r, -1.0, 0.0001);
    EXPECT_NEAR(color.c.g, 203.0, 0.0001);
    EXPECT_NEAR(color.c.b, 12500.0, 0.0001);
    EXPECT_DOUBLE_EQ(color.c.a, 1.0);
}

TEST(ColorManagement, ParametricRangesHonorLuminances) {
    for (auto tf : {CM_TRANSFER_FUNCTION_EXT_LINEAR, CM_TRANSFER_FUNCTION_EXT_SRGB, CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_GAMMA22, CM_TRANSFER_FUNCTION_HLG}) {
        const SImageDescription DESC{
            .transferFunction = tf,
            .luminances       = {.min = 0.01f, .max = 600, .reference = 308},
        };
        EXPECT_FLOAT_EQ(DESC.getTFMinLuminance(), 0.01f);
        EXPECT_FLOAT_EQ(DESC.getTFMaxLuminance(), 600.0f);
    }
}

TEST(ColorManagement, PQEncodingRangeIsIndependentOfContentPeak) {
    const SImageDescription DESC{
        .transferFunction = CM_TRANSFER_FUNCTION_ST2084_PQ,
        .luminances       = {.min = 0.1f, .max = 1000, .reference = 308},
        .maxCLL           = 308,
    };
    EXPECT_FLOAT_EQ(DESC.getTFMinLuminance(), 0.1f);
    EXPECT_FLOAT_EQ(DESC.getTFMaxLuminance(), 0.1f + HDR_MAX_LUMINANCE - HDR_MIN_LUMINANCE);
    EXPECT_FLOAT_EQ(DESC.getContentMaxLuminance(), 308.0f);
}

TEST(ColorManagement, WindowsBT2100UsesHDRLuminances) {
    const auto& DESC = BT2100_IMAGE_DESCRIPTION->value();
    EXPECT_FLOAT_EQ(DESC.getTFMinLuminance(), HDR_MIN_LUMINANCE);
    EXPECT_FLOAT_EQ(DESC.getTFMaxLuminance(), HDR_MAX_LUMINANCE);
    EXPECT_EQ(DESC.luminances.reference, HDR_REF_LUMINANCE);
}

TEST(ColorManagement, ContentPeakPrefersCLLThenMasteringThenEncoding) {
    auto desc                    = DEFAULT_HDR_IMAGE_DESCRIPTION->value();
    desc.masteringLuminances.max = 1000;
    desc.maxCLL                  = 600;
    EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), 600.0f);
    desc.maxCLL = 0;
    EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), 1000.0f);
    desc.maxCLL = 100; // Dark scenes can peak below reference white.
    EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), 100.0f);
    desc.maxCLL = 20000;
    EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), 1000.0f);
    desc.masteringLuminances.max = 0;
    EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), desc.getTFMaxLuminance());
}

TEST(ColorManagement, WorkBufferContentPeakUsesMonitorRange) {
    // The FP16 work buffer shares scRGB's encoding but carries the monitor's range, so it must
    // stay distinct from client scRGB descriptions and report the monitor peak, not the PQ ceiling.
    const SImageDescription::SPCLuminances LUMINANCES{.min = 0.005f, .max = 600, .reference = 203};
    const auto                             WORK  = LINEAR_IMAGE_DESCRIPTION->with(LUMINANCES);
    const auto                             SCRGB = SCRGB_IMAGE_DESCRIPTION->with(LUMINANCES);
    EXPECT_NE(WORK->id(), SCRGB->id());
    EXPECT_FLOAT_EQ(WORK->value().getContentMaxLuminance(), 600.0f);
    EXPECT_FLOAT_EQ(SCRGB->value().getContentMaxLuminance(), HDR_MAX_LUMINANCE);
}

TEST(ColorManagement, SDRRangeOverridesRemainSupported) {
    const auto& DESC = DEFAULT_SRGB_IMAGE_DESCRIPTION->value();
    EXPECT_FLOAT_EQ(DESC.getTFMinLuminance(0.01f), 0.01f);
    EXPECT_FLOAT_EQ(DESC.getTFMaxLuminance(308), 308.0f);
    EXPECT_FLOAT_EQ(DESC.getContentMaxLuminance(), 80.0f);
}

TEST(ColorManagement, GenericLinearDefaultsAreNotWindowsSCRGB) {
    const SImageDescription DESC{.transferFunction = CM_TRANSFER_FUNCTION_EXT_LINEAR};
    EXPECT_FLOAT_EQ(DESC.getDefaultTFMinLuminance(), 0.2f);
    EXPECT_FLOAT_EQ(DESC.getDefaultTFMaxLuminance(), 80.0f);
    EXPECT_FLOAT_EQ(DESC.getTFRefLuminance(), 80.0f);
}

TEST(ColorManagement, ExtendedContentPeakUsesDeclaredRangeWithoutMetadata) {
    for (auto tf : {CM_TRANSFER_FUNCTION_EXT_LINEAR, CM_TRANSFER_FUNCTION_EXT_SRGB}) {
        for (uint32_t peak : {80U, 308U, 1000U}) {
            SImageDescription desc{
                .transferFunction = tf,
                .luminances       = {.min = 0, .max = peak, .reference = 80},
            };
            EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), peak);
            desc.masteringLuminances.max = 2000;
            EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), 2000.0f);
            desc.maxCLL = 1500;
            EXPECT_FLOAT_EQ(desc.getContentMaxLuminance(), 1500.0f);
        }
    }
    EXPECT_FLOAT_EQ(SCRGB_IMAGE_DESCRIPTION->value().getContentMaxLuminance(), 10000.0f);
}
