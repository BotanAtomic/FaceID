# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\botan\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\202.6397.12\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\botan\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\202.6397.12\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\botan\Work\FaceID\project\lib\linear-model

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles\ML-test.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\ML-test.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\ML-test.dir\flags.make

CMakeFiles\ML-test.dir\library.cpp.obj: CMakeFiles\ML-test.dir\flags.make
CMakeFiles\ML-test.dir\library.cpp.obj: ..\library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ML-test.dir/library.cpp.obj"
	C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\ML-test.dir\library.cpp.obj /FdCMakeFiles\ML-test.dir\ /FS -c C:\Users\botan\Work\FaceID\project\lib\linear-model\library.cpp
<<

CMakeFiles\ML-test.dir\library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML-test.dir/library.cpp.i"
	C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\cl.exe > CMakeFiles\ML-test.dir\library.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\botan\Work\FaceID\project\lib\linear-model\library.cpp
<<

CMakeFiles\ML-test.dir\library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML-test.dir/library.cpp.s"
	C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\ML-test.dir\library.cpp.s /c C:\Users\botan\Work\FaceID\project\lib\linear-model\library.cpp
<<

CMakeFiles\ML-test.dir\test.cpp.obj: CMakeFiles\ML-test.dir\flags.make
CMakeFiles\ML-test.dir\test.cpp.obj: ..\test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ML-test.dir/test.cpp.obj"
	C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\ML-test.dir\test.cpp.obj /FdCMakeFiles\ML-test.dir\ /FS -c C:\Users\botan\Work\FaceID\project\lib\linear-model\test.cpp
<<

CMakeFiles\ML-test.dir\test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML-test.dir/test.cpp.i"
	C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\cl.exe > CMakeFiles\ML-test.dir\test.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\botan\Work\FaceID\project\lib\linear-model\test.cpp
<<

CMakeFiles\ML-test.dir\test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML-test.dir/test.cpp.s"
	C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\ML-test.dir\test.cpp.s /c C:\Users\botan\Work\FaceID\project\lib\linear-model\test.cpp
<<

# Object files for target ML-test
ML__test_OBJECTS = \
"CMakeFiles\ML-test.dir\library.cpp.obj" \
"CMakeFiles\ML-test.dir\test.cpp.obj"

# External object files for target ML-test
ML__test_EXTERNAL_OBJECTS =

ML-test.exe: CMakeFiles\ML-test.dir\library.cpp.obj
ML-test.exe: CMakeFiles\ML-test.dir\test.cpp.obj
ML-test.exe: CMakeFiles\ML-test.dir\build.make
ML-test.exe: CMakeFiles\ML-test.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ML-test.exe"
	C:\Users\botan\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\202.6397.12\bin\cmake\win\bin\cmake.exe -E vs_link_exe --intdir=CMakeFiles\ML-test.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MICROS~1\2019\PROFES~1\VC\Tools\MSVC\1426~1.288\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\ML-test.dir\objects1.rsp @<<
 /out:ML-test.exe /implib:ML-test.lib /pdb:C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release\ML-test.pdb /version:0.0  /machine:x64 /INCREMENTAL:NO /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\ML-test.dir\build: ML-test.exe

.PHONY : CMakeFiles\ML-test.dir\build

CMakeFiles\ML-test.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\ML-test.dir\cmake_clean.cmake
.PHONY : CMakeFiles\ML-test.dir\clean

CMakeFiles\ML-test.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\botan\Work\FaceID\project\lib\linear-model C:\Users\botan\Work\FaceID\project\lib\linear-model C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release C:\Users\botan\Work\FaceID\project\lib\linear-model\cmake-build-release\CMakeFiles\ML-test.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\ML-test.dir\depend

