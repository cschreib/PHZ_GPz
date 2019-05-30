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
if (BYPASS_ELEMENTS)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
endif()

find_package(Eigen3 REQUIRED)

if (NOT BYPASS_ELEMENTS OR DO_TEST)
    find_package(Boost COMPONENTS unit_test_framework)
endif()

if (NOT NO_GSL)
    find_package(GSL)
endif()

if (NOT BYPASS_ELEMENTS)
    #===============================================================================
    # Create configuration header
    #===============================================================================
    configure_file(PHZ_GPz/Config.h.in PHZ_GPz/Config.h)

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
                         INCLUDE_DIRS GSL Eigen3 ElementsKernel ${CMAKE_CURRENT_BINARY_DIR}
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

    include_directories(${CMAKE_CURRENT_BINARY_DIR})
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

    install(FILES
        PHZ_GPz/EigenTypes.h
        PHZ_GPz/EigenWrapper.h
        PHZ_GPz/GPz.h
        PHZ_GPz/GSLWrapper.h
        PHZ_GPz/LBFGS.h
        PHZ_GPz/Minimize.h
        PHZ_GPz/STLWrapper.h
        PHZ_GPz/Utils.h
        DESTINATION include/PHZ_GPz)

    configure_file(PHZ_GPz/Config.h.in PHZ_GPz/Config.h)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/PHZ_GPz/Config.h
        DESTINATION include/PHZ_GPz)

    if (DO_TEST)
        enable_testing()
        set(BOOST_INCLUDE_DIRS ${boost_installation_prefix}/include)

        # creates the executable
        add_executable(GPz_test tests/src/GPz_test.cpp)
        target_link_libraries(GPz_test GPz)
        target_link_libraries(GPz_test ${Boost_LIBRARIES})
        target_link_libraries(GPz_test ${GPZ_LIBRARIES})
        target_link_libraries(GPz_test ${EIGEN3_LIBRARIES})
        target_compile_definitions(GPz_test PRIVATE "BOOST_TEST_DYN_LINK=1")
        target_compile_definitions(GPz_test PRIVATE "BOOST_TEST_MODULE=1")
        if (NOT NO_GSL)
            target_link_libraries(GPz_test ${GSL_LIBRARIES})
        endif()
        # indicates the include paths
        target_include_directories(GPz_test PRIVATE ${Boost_INCLUDE_DIRS})

        # declares a test with our executable
        add_test(NAME test COMMAND GPz_test)
    endif()
endif()
