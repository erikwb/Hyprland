#include <helpers/cm/ColorManagement.hpp>
#include <gtest/gtest.h>

using namespace NColorManagement;

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
    EXPECT_FLOAT_EQ(DESC.getTFMaxLuminance(), 10000.1f);
    EXPECT_FLOAT_EQ(DESC.getContentMaxLuminance(), 308.0f);
}

TEST(ColorManagement, WindowsAndInternalLinearRetainEightyNitUnits) {
    for (const auto& DESC : {SCRGB_IMAGE_DESCRIPTION, LINEAR_IMAGE_DESCRIPTION}) {
        const auto CHANGED = DESC->with({.min = 0.01f, .max = 518, .reference = 308});
        EXPECT_FLOAT_EQ(CHANGED->value().getTFMinLuminance(), 0.0f);
        EXPECT_FLOAT_EQ(CHANGED->value().getTFMaxLuminance(), 80.0f);
        auto parametric       = CHANGED->value();
        parametric.isWindows  = false;
        parametric.isInternal = false;
        const auto OTHER      = CImageDescription::from(parametric);
        EXPECT_NE(CHANGED->id(), OTHER->id());
        EXPECT_TRUE(CHANGED->needsCM(OTHER));
        EXPECT_TRUE(OTHER->needsCM(CHANGED));
    }
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
