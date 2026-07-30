foreach(required LLVM_OBJDUMP LLVM_READELF SCALAR_OBJECT AVX2_OBJECT AVX512_OBJECT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

foreach(object_file SCALAR_OBJECT AVX2_OBJECT AVX512_OBJECT)
    if(NOT EXISTS "${${object_file}}")
        message(FATAL_ERROR "object does not exist: ${${object_file}}")
    endif()
endforeach()

function(disassemble object output_variable)
    execute_process(
        COMMAND "${LLVM_OBJDUMP}" -d "${object}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-objdump failed for ${object}: ${error}")
    endif()
    set(${output_variable} "${output}" PARENT_SCOPE)
endfunction()

function(check_hidden_symbols object raw_symbol)
    execute_process(
        COMMAND "${LLVM_READELF}" --symbols "${object}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-readelf failed for ${object}: ${error}")
    endif()
    foreach(symbol "${raw_symbol}" "_mlir_ciface_${raw_symbol}")
        if(NOT output MATCHES "GLOBAL[ \t]+HIDDEN[^\n]*${symbol}")
            message(FATAL_ERROR "${symbol} is not GLOBAL HIDDEN in ${object}")
        endif()
    endforeach()
endfunction()

disassemble("${SCALAR_OBJECT}" scalar_assembly)
disassemble("${AVX2_OBJECT}" avx2_assembly)
disassemble("${AVX512_OBJECT}" avx512_assembly)

check_hidden_symbols("${SCALAR_OBJECT}" "deepforge_conv2d_scalar")
check_hidden_symbols("${AVX2_OBJECT}" "deepforge_conv2d_avx2")
check_hidden_symbols("${AVX512_OBJECT}" "deepforge_conv2d_avx512")

if(scalar_assembly MATCHES "[ \t]v[a-z0-9]+[ \t]")
    message(FATAL_ERROR "scalar object contains a VEX/EVEX instruction")
endif()
if(NOT avx2_assembly MATCHES "ymm")
    message(FATAL_ERROR "AVX2 object does not use YMM registers")
endif()
if(avx2_assembly MATCHES "zmm")
    message(FATAL_ERROR "AVX2 object unexpectedly uses ZMM registers")
endif()
if(NOT avx512_assembly MATCHES "zmm")
    message(FATAL_ERROR "AVX-512 object does not use ZMM registers")
endif()
