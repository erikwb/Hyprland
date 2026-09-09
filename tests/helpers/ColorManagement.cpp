#include <helpers/cm/ColorManagement.hpp>
#include <gtest/gtest.h>

using namespace NColorManagement;

TEST(ColorManagement, ParametricRangesHonorLuminances) {
    for (auto tf : {CM_TRANSFER_FUNCTION_EXT_LINEAR, CM_TRANSFER_FUNCTION_EXT_SRGB, CM_TRANSFER_FUNCTION_SRGB, CM_TRANSFER_FUNCTION_GAMMA22, CM_TRANSFER_FUNCTION_HLG}) {
        const SImageDescription desc{
            .transferFunction = tf,
            .luminances       = {.min = 0.01f, .max = 600, .reference = 308},
        };
        EXPECT_FLOAT_EQ(desc.getTFMinLuminance(), 0.01f);
        EXPECT_FLOAT_EQ(desc.getTFMaxLuminance(), 600.0f);
    }
}

TEST(ColorManagement, PQEncodingRangeIsIndependentOfContentPeak) {
    const SImageDescription desc{
        .transferFunction = CM_TRANSFER_FUNCTION_ST2084_PQ,
        .luminances       = {.min = 0.1f, .max = 1000, .reference = 308},
        .maxCLL           = 308,
    };
    EXPECT_FLOAT_EQ(desc.getTFMinLuminance(), 0.1f);
    EXPECT_FLOAT_EQ(desc.getTFMaxLuminance(), 10000.1f);
}

TEST(ColorManagement, WindowsAndInternalLinearRetainEightyNitUnits) {
    for (const auto desc : {SCRGB_IMAGE_DESCRIPTION, LINEAR_IMAGE_DESCRIPTION}) {
        const auto changed = desc->with({.min = 0.01f, .max = 518, .reference = 308});
        EXPECT_FLOAT_EQ(changed->value().getTFMinLuminance(), 0.0f);
        EXPECT_FLOAT_EQ(changed->value().getTFMaxLuminance(), 80.0f);
        auto parametric       = changed->value();
        parametric.isWindows  = false;
        parametric.isInternal = false;
        const auto other      = CImageDescription::from(parametric);
        EXPECT_NE(changed->id(), other->id());
        EXPECT_TRUE(changed->needsCM(other));
        EXPECT_TRUE(other->needsCM(changed));
    }
}

TEST(ColorManagement, SDRRangeOverridesRemainSupported) {
    const auto& desc = DEFAULT_SRGB_IMAGE_DESCRIPTION->value();
    EXPECT_FLOAT_EQ(desc.getTFMinLuminance(0.01f), 0.01f);
    EXPECT_FLOAT_EQ(desc.getTFMaxLuminance(308), 308.0f);
}

TEST(ColorManagement, GenericLinearDefaultsAreNotWindowsSCRGB) {
    const SImageDescription desc{.transferFunction = CM_TRANSFER_FUNCTION_EXT_LINEAR};
    EXPECT_FLOAT_EQ(desc.getDefaultTFMinLuminance(), 0.2f);
    EXPECT_FLOAT_EQ(desc.getDefaultTFMaxLuminance(), 80.0f);
    EXPECT_FLOAT_EQ(desc.getTFRefLuminance(), 80.0f);
}
