# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/therapy/Astroscanner/workspace/opencv-4.10.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/therapy/Astroscanner/workspace/opencv-4.10.0/release

# Include any dependencies generated for this target.
include modules/freetype/CMakeFiles/opencv_freetype.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include modules/freetype/CMakeFiles/opencv_freetype.dir/compiler_depend.make

# Include the progress variables for this target.
include modules/freetype/CMakeFiles/opencv_freetype.dir/progress.make

# Include the compile flags for this target's objects.
include modules/freetype/CMakeFiles/opencv_freetype.dir/flags.make

modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o: modules/freetype/CMakeFiles/opencv_freetype.dir/flags.make
modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o: /home/therapy/Astroscanner/workspace/opencv_contrib-4.10.0/modules/freetype/src/freetype.cpp
modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o: modules/freetype/CMakeFiles/opencv_freetype.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/therapy/Astroscanner/workspace/opencv-4.10.0/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o"
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o -MF CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o.d -o CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o -c /home/therapy/Astroscanner/workspace/opencv_contrib-4.10.0/modules/freetype/src/freetype.cpp

modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_freetype.dir/src/freetype.cpp.i"
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/therapy/Astroscanner/workspace/opencv_contrib-4.10.0/modules/freetype/src/freetype.cpp > CMakeFiles/opencv_freetype.dir/src/freetype.cpp.i

modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_freetype.dir/src/freetype.cpp.s"
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/therapy/Astroscanner/workspace/opencv_contrib-4.10.0/modules/freetype/src/freetype.cpp -o CMakeFiles/opencv_freetype.dir/src/freetype.cpp.s

# Object files for target opencv_freetype
opencv_freetype_OBJECTS = \
"CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o"

# External object files for target opencv_freetype
opencv_freetype_EXTERNAL_OBJECTS =

lib/libopencv_freetype.so.4.10.0: modules/freetype/CMakeFiles/opencv_freetype.dir/src/freetype.cpp.o
lib/libopencv_freetype.so.4.10.0: modules/freetype/CMakeFiles/opencv_freetype.dir/build.make
lib/libopencv_freetype.so.4.10.0: lib/libopencv_imgproc.so.4.10.0
lib/libopencv_freetype.so.4.10.0: 3rdparty/lib/libtegra_hal.a
lib/libopencv_freetype.so.4.10.0: /usr/lib/aarch64-linux-gnu/libfreetype.so
lib/libopencv_freetype.so.4.10.0: /usr/lib/aarch64-linux-gnu/libharfbuzz.so
lib/libopencv_freetype.so.4.10.0: lib/libopencv_core.so.4.10.0
lib/libopencv_freetype.so.4.10.0: lib/libopencv_cudev.so.4.10.0
lib/libopencv_freetype.so.4.10.0: /usr/lib/aarch64-linux-gnu/libfreetype.so
lib/libopencv_freetype.so.4.10.0: /usr/lib/aarch64-linux-gnu/libharfbuzz.so
lib/libopencv_freetype.so.4.10.0: modules/freetype/CMakeFiles/opencv_freetype.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/therapy/Astroscanner/workspace/opencv-4.10.0/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../lib/libopencv_freetype.so"
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_freetype.dir/link.txt --verbose=$(VERBOSE)
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype && $(CMAKE_COMMAND) -E cmake_symlink_library ../../lib/libopencv_freetype.so.4.10.0 ../../lib/libopencv_freetype.so.410 ../../lib/libopencv_freetype.so

lib/libopencv_freetype.so.410: lib/libopencv_freetype.so.4.10.0
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libopencv_freetype.so.410

lib/libopencv_freetype.so: lib/libopencv_freetype.so.4.10.0
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libopencv_freetype.so

# Rule to build all files generated by this target.
modules/freetype/CMakeFiles/opencv_freetype.dir/build: lib/libopencv_freetype.so
.PHONY : modules/freetype/CMakeFiles/opencv_freetype.dir/build

modules/freetype/CMakeFiles/opencv_freetype.dir/clean:
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype && $(CMAKE_COMMAND) -P CMakeFiles/opencv_freetype.dir/cmake_clean.cmake
.PHONY : modules/freetype/CMakeFiles/opencv_freetype.dir/clean

modules/freetype/CMakeFiles/opencv_freetype.dir/depend:
	cd /home/therapy/Astroscanner/workspace/opencv-4.10.0/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/therapy/Astroscanner/workspace/opencv-4.10.0 /home/therapy/Astroscanner/workspace/opencv_contrib-4.10.0/modules/freetype /home/therapy/Astroscanner/workspace/opencv-4.10.0/release /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype /home/therapy/Astroscanner/workspace/opencv-4.10.0/release/modules/freetype/CMakeFiles/opencv_freetype.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/freetype/CMakeFiles/opencv_freetype.dir/depend
