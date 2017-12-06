@echo off
pushd misc
    call configEnv
popd

set entryFileName=%1
set sourceDir=%2
set outputName=%3
set outputDir=%4
set buildType=%5

set entryFile=%sourceDir%\%entryFileName%

mkdir %outputDir%\%buildType%

pushd %outputDir%\%buildType%
    set compilerflags=/Ot /EHsc /std:c++latest
    if %buildType% == debug (
        set compilerflags=%compilerflags% /Zi
    )
    if %buildType% == release (
        set compilerflags=%compilerflags% /O2
    )
    set linkerflags=/OUT:%outputName%

    cl.exe %compilerflags% %entryFile% /link %linkerflags%
popd