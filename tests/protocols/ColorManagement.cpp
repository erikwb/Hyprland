#include <protocols/ColorManagement.hpp>
#include <protocols/PresentationTime.hpp>
#include <protocols/core/Compositor.hpp>
#include <protocols/core/Subcompositor.hpp>
#include <config/ConfigValue.hpp>
#include <config/lua/ConfigManager.hpp>
#include <config/shared/inotify/ConfigWatcher.hpp>
#include <Compositor.hpp>
#include <event/EventBus.hpp>

#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <sys/socket.h>

using namespace Hyprutils::OS;

class CColorFeedbackTest : public testing::Test {
  protected:
    void                                                       SetUp() override;
    void                                                       TearDown() override;
    SP<CWLSurfaceResource>                                     surface();
    SP<CWLSubsurfaceResource>                                  subsurface(SP<CWLSurfaceResource> child, SP<CWLSurfaceResource> parent);

    std::unique_ptr<wl_display, decltype(&wl_display_destroy)> m_display{nullptr, wl_display_destroy};
    CFileDescriptor                                            m_socket;
    wl_client*                                                 m_client = nullptr;
    uint32_t                                                   m_nextID = 2;

  private:
    UP<CCompositor>            m_previousCompositor;
    UP<Config::IConfigManager> m_previousConfig;
    UP<Config::CConfigWatcher> m_previousWatcher;
    UP<Event::CEventBus>       m_previousBus;
    UP<CPresentationProtocol>  m_previousPresentation;
};

void CColorFeedbackTest::SetUp() {
    m_previousCompositor   = std::move(g_pCompositor);
    m_previousConfig       = std::move(Config::mgr());
    m_previousWatcher      = std::move(Config::watcher());
    m_previousBus          = std::move(Event::bus());
    m_previousPresentation = std::move(PROTO::presentation);

    Event::bus()      = makeUnique<Event::CEventBus>();
    g_pCompositor     = makeUnique<CCompositor>(true);
    Config::watcher() = makeUnique<Config::CConfigWatcher>();
    Config::mgr()     = makeUnique<Config::Lua::CConfigManager>();
    CConfigValueBase::flushCaches();
    m_display.reset(wl_display_create());
    ASSERT_NE(m_display, nullptr);
    g_pCompositor->m_wlDisplay = m_display.get();
    PROTO::presentation        = makeUnique<CPresentationProtocol>(&wp_presentation_interface, 1, "presentation-test");

    std::array<int, 2> sockets = {-1, -1};
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.data()), 0);
    CFileDescriptor serverSocket{sockets.at(0)};
    m_socket = CFileDescriptor{sockets.at(1)};
    m_client = wl_client_create(m_display.get(), serverSocket.get());
    ASSERT_NE(m_client, nullptr);
    serverSocket.take();
}

void CColorFeedbackTest::TearDown() {
    if (m_display)
        wl_display_destroy_clients(m_display.get());
    PROTO::presentation = std::move(m_previousPresentation);
    Config::mgr()       = std::move(m_previousConfig);
    if (Config::mgr())
        CConfigValueBase::flushCaches();
    Config::watcher() = std::move(m_previousWatcher);
    g_pCompositor     = std::move(m_previousCompositor);
    Event::bus()      = std::move(m_previousBus);
}

SP<CWLSurfaceResource> CColorFeedbackTest::surface() {
    auto wire      = makeShared<CWlSurface>(m_client, 6, m_nextID++);
    auto result    = makeShared<CWLSurfaceResource>(wire);
    result->m_self = result;
    // The fixture owns these resources instead of the global protocol managers.
    wire->setOnDestroy([](CWlSurface*) { ; });
    return result;
}

SP<CWLSubsurfaceResource> CColorFeedbackTest::subsurface(SP<CWLSurfaceResource> child, SP<CWLSurfaceResource> parent) {
    auto wire      = makeShared<CWlSubsurface>(m_client, 1, m_nextID++);
    auto result    = makeShared<CWLSubsurfaceResource>(wire, child, parent);
    result->m_self = result;
    child->m_role  = makeShared<CSubsurfaceRole>(result);
    wire->setOnDestroy([](CWlSubsurface*) { ; });
    return result;
}

TEST_F(CColorFeedbackTest, DestroyedParentDoesNotCrashPreferredFeedback) {
    auto       parent   = surface();
    const auto CHILD    = surface();
    const auto SUB      = subsurface(CHILD, parent);
    auto       wire     = makeShared<CWpColorManagementSurfaceFeedbackV1>(m_client, 2, m_nextID++);
    const auto FEEDBACK = makeShared<CColorManagementFeedbackSurface>(wire, CHILD);
    wire->setOnDestroy([](CWpColorManagementSurfaceFeedbackV1*) { ; });

    parent.reset();
    EXPECT_TRUE(SUB->m_parent.expired());
    EXPECT_FALSE(SUB->t1Parent());
    EXPECT_EQ(CHILD->getPreferredImageDescription(), g_pCompositor->getPreferredImageDescription());
    // Enter/leave and monitor color changes use the same preferred-change callback.
    CHILD->m_events.leave.emit(PHLMONITOR{});
}

TEST_F(CColorFeedbackTest, NestedSubsurfaceFindsRootAndHandlesExpiredAncestor) {
    auto       root       = surface();
    const auto FIRST      = surface();
    const auto SECOND     = surface();
    const auto CHILD      = surface();
    const auto FIRST_SUB  = subsurface(FIRST, root);
    const auto SECOND_SUB = subsurface(SECOND, FIRST);
    const auto CHILD_SUB  = subsurface(CHILD, SECOND);
    EXPECT_EQ(CHILD_SUB->t1Parent(), root);

    root.reset();
    EXPECT_FALSE(CHILD_SUB->t1Parent());
    EXPECT_EQ(CHILD->getPreferredImageDescription(), g_pCompositor->getPreferredImageDescription());
}
