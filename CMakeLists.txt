CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

if (NOT BYPASS_ELEMENTS)
    #===============================================================================
    # Load macros and functions for Elements-based projects
    #===============================================================================

    find_package(ElementsProject)

    #===============================================================================
    # Declare project name and version
    # Example with dependency:
    #                         elements_project(MyProject 1.0 USE Element 3.9)
    #===============================================================================

    elements_project(PHZ_GPz 0.1 USE Elements 5.4)
else()
    #===============================================================================
    # Compile as standalone project, outside of Elements
    #===============================================================================

    if (NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, default to Release (-DCMAKE_BUILD_TYPE=...)")
        set(CMAKE_BUILD_TYPE "Release")
    endif()

    add_subdirectory(PHZ_GPz)
endif()
