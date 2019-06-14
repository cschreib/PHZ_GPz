# - Try to find GPz lib
#
# Once done this will define
#
#  GPZ_FOUND - system has GPz lib
#  GPZ_INCLUDE_DIR - the GPz include directory
#  GPZ_VERSION - GPz version
#
# This module reads hints about search locations from
# the following enviroment variables:
#
# GPZ_ROOT_DIR

if (NOT GPz_FIND_VERSION)
    if(NOT GPz_FIND_VERSION_MAJOR)
        set(GPz_FIND_VERSION_MAJOR 0)
    endif(NOT GPz_FIND_VERSION_MAJOR)
    if(NOT GPz_FIND_VERSION_MINOR)
        set(GPz_FIND_VERSION_MINOR 9)
    endif(NOT GPz_FIND_VERSION_MINOR)
    if(NOT GPz_FIND_VERSION_PATCH)
        set(GPz_FIND_VERSION_PATCH 9)
    endif(NOT GPz_FIND_VERSION_PATCH)

    set(GPz_FIND_VERSION "${GPz_FIND_VERSION_MAJOR}.${GPz_FIND_VERSION_MINOR}.${GPz_FIND_VERSION_PATCH}")
endif()

macro(_gpz_check_version)
    file(READ "${GPZ_INCLUDE_DIR}/PHZ_GPz/Config.h" _gpz_version_header)

    string(REGEX MATCH "define[ \t]+GPZ_WORLD_VERSION[ \t]+([0-9]+)" _gpz_world_version_match "${_gpz_version_header}")
    set(GPZ_WORLD_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+GPZ_MAJOR_VERSION[ \t]+([0-9]+)" _gpz_major_version_match "${_gpz_version_header}")
    set(GPZ_MAJOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+GPZ_MINOR_VERSION[ \t]+([0-9]+)" _gpz_minor_version_match "${_gpz_version_header}")
    set(GPZ_MINOR_VERSION "${CMAKE_MATCH_1}")

    set(GPZ_VERSION ${GPZ_WORLD_VERSION}.${GPZ_MAJOR_VERSION}.${GPZ_MINOR_VERSION})
    if(${GPZ_VERSION} VERSION_LESS ${GPz_FIND_VERSION})
        set(GPZ_VERSION_OK FALSE)
    else()
        set(GPZ_VERSION_OK TRUE)
    endif()

    if(NOT GPZ_VERSION_OK)
        message(STATUS "GPz version ${GPZ_VERSION} found in ${GPZ_INCLUDE_DIR}, "
                   "but at least version ${GPz_FIND_VERSION} is required")
    endif()
endmacro()

if (GPZ_INCLUDE_DIR)

    # in cache already
    _gpz_check_version()
    set(GPZ_FOUND ${GPZ_VERSION_OK})
    set(GPZ_INCLUDE_DIRS ${GPZ_INCLUDE_DIR})
    list(APPEND GPZ_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})


else()

    # search first if an GPzConfig.cmake is available in the system,
    # if successful this would set GPZ_INCLUDE_DIR and the rest of
    # the script will work as usual
    if (NOT GPZ_ROOT_DIR)
        find_package(GPz ${GPz_FIND_VERSION} NO_MODULE QUIET)
    endif()

    if (NOT GPZ_INCLUDE_DIR)
        find_path(GPZ_INCLUDE_DIR NAMES PHZ_GPz/GPz.h
            HINTS
            ENV GPZ_ROOT_DIR
            ${GPZ_ROOT_DIR}
        )
    endif()

    if (NOT GPZ_LIBRARIES)
        find_library(GPZ_LIBRARIES GPz
            HINTS ${GPZ_ROOT_DIR}
            PATH_SUFFIXES lib
        )
    endif()

    if(GPZ_INCLUDE_DIR)
        _gpz_check_version()
    endif()

    find_package(Eigen3 REQUIRED)

    set(GPZ_INCLUDE_DIRS ${GPZ_INCLUDE_DIR})
    list(APPEND GPZ_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(GPz DEFAULT_MSG
        GPZ_INCLUDE_DIR GPZ_INCLUDE_DIRS GPZ_LIBRARIES GPZ_VERSION_OK)

    mark_as_advanced(GPZ_INCLUDE_DIR GPZ_INCLUDE_DIRS)

endif()

