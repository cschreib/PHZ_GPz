CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

if (NOT BYPASS_ELEMENTS)
    #===============================================================================
    # Load elements_subdir macro here
    # Examples:
    #   For declaring a project module:
    #         elements_subdir(ElementsExamples)
    #===============================================================================
    elements_subdir(PHZ_GPz)

    #===============================================================================
    # Load elements_depends_on_subdirs macro here
    #   For creating a dependency onto an other accessible module
    #         elements_depends_on_subdirs(ElementsKernel)
    #===============================================================================
    elements_depends_on_subdirs(ElementsKernel)
endif()

#===============================================================================
# Add the find_package macro (a pure CMake command) here to locate the
# libraries.
# Examples:
#          find_package(CppUnit)
#===============================================================================
find_package(Eigen3 REQUIRED)
find_package(Boost)

if (NOT NO_GSL)
    find_package(GSL)
endif()

if (NOT BYPASS_ELEMENTS)
    #===============================================================================
    # Declare the library dependencies here
    # Example:
    #         elements_add_library(ElementsExamples src/Lib/*.cpp
    #                     LINK_LIBRARIES Boost ElementsKernel
    #                     INCLUDE_DIRS Boost ElementsKernel
    #                     PUBLIC_HEADERS ElementsExamples)
    #===============================================================================
    elements_add_library(PHZ_GPz src/lib/*.cpp
                         LINK_LIBRARIES GSL Eigen3 ElementsKernel
                         INCLUDE_DIRS GSL Eigen3 ElementsKernel
                         PUBLIC_HEADERS PHZ_GPz)

    #===============================================================================
    # Declare the executables here
    # Example:
    # elements_add_executable(ElementsProgramExample src/Program/ProgramExample.cpp
    #                        LINK_LIBRARIES Boost ElementsExamples
    #                        INCLUDE_DIRS Boost ElementsExamples)
    #===============================================================================

    #===============================================================================
    # Declare the Boost tests here
    # Example:
    # elements_add_unit_test(BoostClassExample tests/src/Boost/ClassExample_test.cpp
    #                       EXECUTABLE BoostClassExample_test
    #                       INCLUDE_DIRS ElementsExamples
    #                       LINK_LIBRARIES ElementsExamples TYPE Boost)
    #===============================================================================
    elements_add_unit_test(GPz tests/src/GPz_test.cpp
                         EXECUTABLE PHZ_GPz_GPz_test
                         LINK_LIBRARIES PHZ_GPz
                         TYPE Boost)

    #===============================================================================
    # Use the following macro for python modules, scripts and aux files:
    #  elements_install_python_modules()
    #  elements_install_scripts()
    #===============================================================================
    elements_install_python_modules()

    #===============================================================================
    # Declare the Python programs here
    # Examples :
    # elements_add_python_program(PythonProgramExample
    #                             ElementsExamples.PythonProgramExample)
    #===============================================================================

    #===============================================================================
    # Add the elements_install_conf_files macro
    # Examples:
    #          elements_install_conf_files()
    #===============================================================================
else()
    #===============================================================================
    # Compile as standalone project, outside of Elements
    #===============================================================================

    if (CMAKE_VERSION VERSION_LESS "3.1")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            add_definitions(-std=c++11)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            add_definitions(-std=c++11)
            add_definitions(-Wall)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            add_definitions(-std=c++11)
            add_definitions(-Wall)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
            # Nothing to do
        endif()
    else()
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()

    include_directories(${PROJECT_SOURCE_DIR}/PHZ_GPz)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    if (NOT NO_GSL)
        include_directories(${GSL_INCLUDE_DIRS})
    endif()

    add_library(GPz SHARED src/lib/GPz.cpp)
    target_link_libraries(GPz ${EIGEN3_LIBRARIES})
    if (NOT NO_GSL)
        target_link_libraries(GPz ${GSL_LIBRARIES})
    endif()

    install(TARGETS GPz DESTINATION lib)

    file(GLOB HEADERS PHZ_GPz/*.h)
    install(FILES ${HEADERS} DESTINATION include/PHZ_GPz)
endif()
