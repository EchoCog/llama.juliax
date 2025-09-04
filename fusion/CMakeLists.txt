# Fusion Language Integration with llama.cpp
# This CMakeLists.txt adds support for J, Julia, and JAX bindings

cmake_minimum_required(VERSION 3.14)

# Fusion language components
option(FUSION_ENABLE_JULIA "Enable Julia bindings" ON)
option(FUSION_ENABLE_J "Enable J language interface" ON) 
option(FUSION_ENABLE_JAX "Enable JAX integration" ON)

if(FUSION_ENABLE_JULIA)
    # Find Julia installation
    find_program(JULIA_EXECUTABLE julia DOC "Julia executable")
    
    if(JULIA_EXECUTABLE)
        message(STATUS "Found Julia: ${JULIA_EXECUTABLE}")
        
        # Add Julia bindings build target
        add_custom_target(fusion-julia
            COMMAND ${JULIA_EXECUTABLE} --project=${CMAKE_CURRENT_SOURCE_DIR}/bindings/julia -e "using Pkg; Pkg.instantiate(); Pkg.build()"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Building Julia fusion bindings"
            VERBATIM
        )
        
        # Add Julia tests target
        add_custom_target(fusion-julia-test
            COMMAND ${JULIA_EXECUTABLE} --project=${CMAKE_CURRENT_SOURCE_DIR}/bindings/julia -e "using Pkg; Pkg.test()"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running Julia fusion tests"
            VERBATIM
            DEPENDS fusion-julia
        )
        
        message(STATUS "Julia fusion bindings enabled")
    else()
        message(WARNING "Julia executable not found, Julia bindings disabled")
        set(FUSION_ENABLE_JULIA OFF)
    endif()
endif()

if(FUSION_ENABLE_J)
    # Find J installation
    find_program(J_EXECUTABLE jconsole DOC "J console executable")
    
    if(J_EXECUTABLE)
        message(STATUS "Found J: ${J_EXECUTABLE}")
        
        # Add J interface test target
        add_custom_target(fusion-j-test
            COMMAND ${J_EXECUTABLE} < ${CMAKE_CURRENT_SOURCE_DIR}/bindings/j/test/test_fusion.ijs
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running J fusion interface tests"
            VERBATIM
        )
        
        message(STATUS "J fusion interface enabled")
    else()
        message(WARNING "J executable not found, J interface disabled")
        set(FUSION_ENABLE_J OFF)
    endif()
endif()

if(FUSION_ENABLE_JAX)
    # Find Python with JAX
    find_program(PYTHON_EXECUTABLE python3 DOC "Python 3 executable")
    
    if(PYTHON_EXECUTABLE)
        # Check if JAX is available
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import jax; print('JAX version:', jax.__version__)"
            RESULT_VARIABLE JAX_CHECK_RESULT
            OUTPUT_VARIABLE JAX_VERSION_OUTPUT
            ERROR_QUIET
        )
        
        if(JAX_CHECK_RESULT EQUAL 0)
            message(STATUS "Found JAX: ${JAX_VERSION_OUTPUT}")
            
            # Add JAX integration build target
            add_custom_target(fusion-jax
                COMMAND ${PYTHON_EXECUTABLE} -m pip install -e ${CMAKE_CURRENT_SOURCE_DIR}/bindings/jax
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Installing JAX fusion bindings"
                VERBATIM
            )
            
            # Add JAX tests target
            add_custom_target(fusion-jax-test
                COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/bindings/jax/test/
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Running JAX fusion tests"
                VERBATIM
                DEPENDS fusion-jax
            )
            
            message(STATUS "JAX fusion integration enabled")
        else()
            message(WARNING "JAX not found in Python environment, JAX integration disabled")
            set(FUSION_ENABLE_JAX OFF)
        endif()
    else()
        message(WARNING "Python executable not found, JAX integration disabled")
        set(FUSION_ENABLE_JAX OFF)
    endif()
endif()

# Create fusion examples target
add_custom_target(fusion-examples
    COMMAND echo "Fusion language examples available in examples/fusion/"
    COMMENT "Fusion language examples"
    VERBATIM
)

# Add comprehensive fusion test target
if(FUSION_ENABLE_JULIA OR FUSION_ENABLE_J OR FUSION_ENABLE_JAX)
    add_custom_target(fusion-test
        COMMENT "Running all fusion language tests"
        VERBATIM
    )
    
    if(FUSION_ENABLE_JULIA)
        add_dependencies(fusion-test fusion-julia-test)
    endif()
    
    if(FUSION_ENABLE_J)
        add_dependencies(fusion-test fusion-j-test)
    endif()
    
    if(FUSION_ENABLE_JAX)
        add_dependencies(fusion-test fusion-jax-test)
    endif()
    
    message(STATUS "Fusion language comprehensive test target created")
endif()

# Summary
message(STATUS "Fusion Language Configuration:")
message(STATUS "  Julia bindings: ${FUSION_ENABLE_JULIA}")
message(STATUS "  J interface: ${FUSION_ENABLE_J}")  
message(STATUS "  JAX integration: ${FUSION_ENABLE_JAX}")

if(FUSION_ENABLE_JULIA OR FUSION_ENABLE_J OR FUSION_ENABLE_JAX)
    message(STATUS "Fusion language features enabled. Use 'make fusion-test' to run all tests.")
else()
    message(STATUS "No fusion language features enabled. Install Julia, J, or JAX to enable bindings.")
endif()