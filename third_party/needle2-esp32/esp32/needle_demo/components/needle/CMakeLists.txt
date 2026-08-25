# Wraps the portable C99 engine (repo-root engine/) as an IDF component so
# esp32/needle_demo can REQUIRE it, exactly as host/CMakeLists.txt links it.
# The engine has no dependencies beyond libm and detects ESP-RTOS via
# ESP_PLATFORM to route allocations to PSRAM-capable heaps.
set(ENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../engine")

file(GLOB ENGINE_SRC "${ENGINE_DIR}/src/*.c")

idf_component_register(SRCS "${ENGINE_SRC}"
                       INCLUDE_DIRS "${ENGINE_DIR}/include")
